# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Cooperative FP4 projection parity and timing on SM120."""

import argparse
import json
from pathlib import Path

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from native_fp4 import FusedGEMV, fused_project, project
from sm120_microbench import timing
from trellis_gemm import make_pair_lut

parser = argparse.ArgumentParser()
parser.add_argument("--parity-only", action="store_true")
parser.add_argument("--cta", action="store_true")
args = parser.parse_args()
torch.manual_seed(617)
lut = make_pair_lut("cuda")
report = {"gpu": torch.cuda.get_device_name(), "rows": []}
for k, n in [
    (128, 128),
    (384, 256),
    (6144, 512),
    (512, 6144),
    (6144, 2048),
    (2048, 6144),
]:
    p = torch.randint(
        -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
    )
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    su = torch.randn(k, device="cuda", dtype=torch.float16)
    sv = torch.randn(n, device="cuda", dtype=torch.float16)
    rows = torch.arange(n, device="cuda")
    o = LinearEXL3(
        None, k, n, trellis=p, suh=su, svh=sv, mul1=torch.tensor(True, device="cuda")
    )
    official = (
        None
        if args.parity_only
        else timing(
            lambda: o.forward(x, {"reconstruct": False}, out_dtype=torch.float32)
        )
    )
    for s in (1, 3, 8, 16, 32, 64, 96):
        if s > k // 64:
            continue
        ref = project(
            x,
            p,
            lut,
            splits=s,
            rows=rows,
            scales=sv,
            input_scales=su,
            arvq=True,
            gemv=True,
        )
        workspace = FusedGEMV(p, lut, su, sv, s) if args.cta else None
        for blocks in (0,) if args.cta else (32, 64, 128, 0):
            call = (
                (lambda: workspace(x))
                if args.cta
                else (lambda: fused_project(x, p, lut, su, sv, s, blocks))
            )
            if args.cta and args.parity_only:
                for _ in range(4):
                    changed = torch.randn_like(x)
                    expected = project(
                        changed,
                        p,
                        lut,
                        splits=s,
                        rows=rows,
                        scales=sv,
                        input_scales=su,
                        arvq=True,
                        gemv=True,
                    )
                    actual = workspace(changed)
                    assert ((actual - expected).norm() / expected.norm()).item() < 1e-5
                    assert torch.count_nonzero(workspace.counters).item() == 0
            out = call()
            rel = ((out - ref).norm() / ref.norm()).item()
            maximum = (out - ref).abs().max().item()
            assert rel < 1e-5, (k, n, s, blocks, rel, maximum)
            row = dict(
                k=k, n=n, splits=s, blocks=blocks, relative_l2=rel, max_abs=maximum
            )
            if not args.parity_only:
                row.update(fused_us=timing(call), official_us=official)
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
if not args.parity_only:
    Path(
        "local-results/sm120_cta_fused_gemv.json"
        if args.cta
        else "local-results/sm120_fused_gemv.json"
    ).write_text(json.dumps(report, indent=2))
print("Fused parity passed", flush=True)
