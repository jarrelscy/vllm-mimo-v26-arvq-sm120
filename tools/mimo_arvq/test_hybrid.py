# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid allocation, packed-weight and frozen-contribution regressions."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "recipe"))
from arvq88.activation import activation_ste
from hybrid import decode, export_hot, hot_output, load_hot, quantize, quantized_expert
from prepare_hybrid import select_hot
from propagation_math import expert_from_packed_input


def test_global_budget_and_cap():
    scores = {
        layer: [float(384 - e) / layer for e in range(384)] for layer in range(1, 70)
    }
    result = select_hot(scores, 1325)
    assert sum(map(len, result.values())) == 1325
    assert max(map(len, result.values())) <= 192
    assert len(result["1"]) > len(result["69"])
    assert result == select_hot(scores, 1325)


def test_nvfp4_pack_roundtrip():
    weights = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )[None]
    packed, scales, global_scale = quantize(weights)
    actual = decode(packed, scales, global_scale)
    torch.testing.assert_close(actual, weights)
    assert torch.equal(
        packed[0, :4], torch.tensor([16, 50, 84, 118], dtype=torch.uint8)
    )
    assert torch.isfinite(decode(*quantize(torch.zeros(2, 128)))).all()


def test_hot_export_and_frozen_contribution(tmp_path):
    class Source:
        def expert(self, layer, expert, projection, device):
            gen = torch.Generator().manual_seed(expert + len(projection))
            return torch.randn(128, 128, generator=gen, device=device) * 0.02

    source = Source()
    (tmp_path / "hot").mkdir()
    (tmp_path / "allocation.json").write_text(
        json.dumps({"layers": {"1": {"hot": [2, 7]}}})
    )
    export_hot(source, 1, [2, 7], tmp_path / "hot/layer1.safetensors", "cpu")
    loaded = load_hot(tmp_path, 1, "cpu")
    for expert in (2, 7):
        _, expected = quantized_expert(source, 1, expert, "cpu")
        for actual, ref in zip(loaded[expert], expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    x = activation_ste(torch.randn(32, 128))
    ids = torch.tensor([[2, 7]] * 32)
    gates = torch.tensor([[0.3, 0.7]] * 32)
    frozen = hot_output(x, ids, gates, loaded)
    expected = sum(
        expert_from_packed_input(x, *loaded[e]) * g for e, g in ((2, 0.3), (7, 0.7))
    )
    torch.testing.assert_close(frozen, expected, rtol=0, atol=0)
    target = torch.randn_like(frozen)
    torch.testing.assert_close((target - frozen) + frozen, target)
