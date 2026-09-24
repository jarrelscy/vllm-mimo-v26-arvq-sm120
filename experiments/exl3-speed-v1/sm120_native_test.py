# ruff: noqa: B023
# Benchmark callbacks execute synchronously within each loop iteration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MMA parity against independently reconstructed FP4 operands."""

import argparse
import json
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from native_fp4 import pack, project
from sm120_microbench import timing
from trellis_gemm import make_pair_lut


def activation_reference(x):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    group = 16 if args.arvq else 32
    residual = x.float().reshape(-1, group)
    result = torch.zeros_like(residual)
    for plane in range(4 if args.arvq else 2):
        scale = 2 ** torch.ceil(
            torch.log2((residual.abs().amax(-1, keepdim=True) / 6).clamp_min(2**-24))
        )
        if args.arvq:
            scale = scale.clamp(2**-6, 2**8)
        index = ((residual.abs() / scale)[..., None] - levels).abs().argmin(-1)
        value = levels[index] * scale * residual.sign()
        result += value / (16**plane if args.arvq else 1)
        residual -= value
        if args.arvq:
            residual *= 16
    return result.reshape_as(x)


parser = argparse.ArgumentParser()
parser.add_argument("--fused", action="store_true")
parser.add_argument("--arvq", action="store_true")
parser.add_argument("--parity-only", action="store_true")
args = parser.parse_args()
result_path = Path(
    "local-results/sm120_native_fused.json"
    if args.fused
    else "local-results/sm120_native.json"
)
if args.arvq:
    result_path = Path("local-results/sm120_native_arvq.json")
torch.manual_seed(91426)
torch.set_num_threads(4)
lut = make_pair_lut("cuda")
levels = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
)
pairs = (levels[:, None] + levels[None, :] / 16).flatten()
checks = []
for rate in (1.5, 2, 2.5):
    k, n = 256, 256
    p = torch.randint(
        -32768, 32767, (16, 16, int(rate * 16)), device="cuda", dtype=torch.int16
    )
    obj = LinearEXL3(
        None,
        k,
        n,
        trellis=p,
        suh=torch.ones(k, device="cuda", dtype=torch.float16),
        svh=torch.ones(n, device="cuda", dtype=torch.float16),
        mul1=torch.tensor(True, device="cuda"),
    )
    w = obj.get_inner_weight_tensor().float()
    w = pairs[(w[..., None] - pairs).abs().argmin(-1)]
    identity = torch.eye(k, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(
        project(identity, p, lut, 2, arvq=args.arvq), w, rtol=0, atol=0
    )
    for m in (1, 4, 7, 8, 9, 16, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        x *= torch.logspace(-3, 2, k // 32, device="cuda").repeat_interleave(32).half()
        reference = activation_reference(x) @ w
        result = project(x, p, lut, 2, arvq=args.arvq)
        rel = ((result - reference).norm() / reference.norm()).item()
        maximum = (result - reference).abs().max().item()
        assert rel < 1e-5, (rate, m, rel)
        checks.append(dict(rate=rate, m=m, relative_l2=rel, max_abs=maximum))
# Exercise partial output tiles and uneven split-K ranges.
for n_tail in (16, 48, 80, 192):
    p_tail = p[:, : n_tail // 16].contiguous()
    x = torch.randn(9, 256, device="cuda", dtype=torch.float16)
    reference = activation_reference(x) @ w[:, :n_tail]
    actual = project(x, p_tail, lut, 3, arvq=args.arvq)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-5)
for m in (1, 9, 64):
    x = torch.randn(m, 6144, device="cuda", dtype=torch.float16)
    su = torch.rand(6144, device="cuda", dtype=torch.float16)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, su, None, 1.0)
    q, sc = pack(xh, arvq=args.arvq)
    fused_q, fused_sc = pack(x, su, arvq=args.arvq)
    torch.testing.assert_close(q, fused_q, rtol=0, atol=0)
    torch.testing.assert_close(sc, fused_sc, rtol=0, atol=0)
print("PARITY", json.dumps(checks), flush=True)
report = {"gpu": torch.cuda.get_device_name(), "parity": checks, "timings": []}
result_path.write_text(json.dumps(report, indent=2))
if args.parity_only:
    raise SystemExit(0)
for k, n in ((6144, 512), (512, 6144), (6144, 2048), (2048, 6144)):
    p = torch.randint(
        -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
    )
    su, sv = (
        torch.ones(k, device="cuda", dtype=torch.float16),
        torch.ones(n, device="cuda", dtype=torch.float16),
    )
    rows = torch.arange(n, device="cuda")
    for m in (1, 2, 4, 16, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        xh = torch.empty_like(x)

        def complete(splits):
            if args.fused:
                return project(
                    x, p, lut, splits, rows, sv, input_scales=su, arvq=args.arvq
                )
            ext.had_r_128(x, xh, su, None, 1.0)
            return project(xh, p, lut, splits, rows, sv, arvq=args.arvq)

        for splits in (1, 2, 4, 8, 16, 32, 64, 96):
            if splits > k // 64:
                continue
            row = dict(
                m=m,
                k=k,
                n=n,
                splits=splits,
                native_projection_us=timing(lambda: complete(splits)),
            )
            report["timings"].append(row)
            print(json.dumps(row), flush=True)
            result_path.write_text(json.dumps(report, indent=2))
