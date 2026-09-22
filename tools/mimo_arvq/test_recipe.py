# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and distributed regressions for the pinned fitting recipe."""

import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent / "recipe"))
from arvq88.activation import ARITHMETIC, activation_ste
from arvq88.encoder import fit_layer_projection, refit_scales, sweep_expert
from arvq88.gradient_indices import expert_output
from arvq88.perf.distributed_batch import broadcast_batch
from propagation_math import expert_from_packed_input


def test_scale_refit_saturates_fp8_overflow():
    weights = torch.full((1, 128), 500.0)
    codes = torch.zeros((1, 16), dtype=torch.uint8)
    scales = refit_scales(
        weights,
        codes,
        codes,
        torch.ones(256, 8),
        torch.zeros(256, 8),
        1.0,
        torch.ones(128),
        128,
    )
    assert scales.item() == 448.0


def test_hessian_factor_keeps_fp32_range():
    torch.manual_seed(81)
    weights = torch.randn(32, 128)
    # Inverse-Cholesky values are valid FP32 but would become zero in FP16.
    hessian = torch.eye(128) * 1e20
    result = fit_layer_projection(
        [weights],
        [hessian],
        device="cpu",
        cb_iters=2,
        sweep_passes=1,
        subsample_per_expert=512,
    )
    assert math.isfinite(result.recon_rel_fro)
    assert torch.isfinite(result.s).all()


def test_triangular_feedback_matches_explicit_inverse():
    torch.manual_seed(82)
    weights = torch.randn(8, 256)
    upper = torch.triu(torch.randn(256, 256) * 0.005) + torch.eye(256)
    books = [torch.randn(256, 8) for _ in range(2)]
    scales = torch.ones(8, 2)
    a, b, actual = sweep_expert(
        weights,
        upper,
        *books,
        scales,
        1.0,
        col_block=128,
        refine=1,
    )
    first = actual[:, :128]
    feedback = (weights[:, :128] - first) @ torch.linalg.inv(upper[:128, :128])
    expected_input = weights[:, 128:] - feedback @ upper[:128, 128:]
    _, _, expected = sweep_expert(
        expected_input,
        upper[128:, 128:],
        *books,
        scales[:, 1:],
        1.0,
        col_block=128,
        refine=1,
    )
    torch.testing.assert_close(actual[:, 128:], expected)
    assert a.shape == b.shape == (8, 32)


def _broadcast_worker(rank, path):
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2
    )
    data = {
        "x": torch.ones(3, 4),
        "required": torch.zeros(3, 4),
        "topk_ids": torch.ones(3, 2, dtype=torch.int64),
        "topk_weights": torch.ones(3, 2),
        "row_weight": torch.tensor([1.0, 50.0, 1.0]),
    }
    result = broadcast_batch(data if rank == 0 else None, "cpu")
    torch.testing.assert_close(result["row_weight"], data["row_weight"])
    assert result["topk_ids"].dtype == torch.int64
    dist.destroy_process_group()


def test_distributed_boundary_weights(tmp_path):
    mp.spawn(_broadcast_worker, args=(str(tmp_path / "gloo"),), nprocs=2)


def test_shared_input_quantization_is_exact():
    torch.manual_seed(83)
    x = torch.randn(128, 128)
    w13, w2 = torch.randn(256, 128) * 0.01, torch.randn(128, 128) * 0.01
    packed = activation_ste(x)
    for rows in (torch.arange(0, 128, 3), torch.tensor([4, 1, 100, 0])):
        actual = expert_from_packed_input(packed[rows], w13, w2)
        expected = expert_output(x[rows], w13, w2, ARITHMETIC)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
