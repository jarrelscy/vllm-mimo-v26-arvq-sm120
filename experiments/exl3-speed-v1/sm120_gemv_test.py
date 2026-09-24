# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Rate-2 batch-one FP4 specialization parity and paired benchmarks."""

import argparse
import json
from pathlib import Path

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from native_fp4 import project
from sm120_microbench import timing
from trellis_gemm import make_pair_lut

parser = argparse.ArgumentParser()
parser.add_argument("--parity-only", action="store_true")
args = parser.parse_args()
torch.manual_seed(42)
lut = make_pair_lut("cuda")
report = {"gpu": torch.cuda.get_device_name(), "rows": []}
for k, n in [
    (128, 16),
    (256, 48),
    (6144, 512),
    (512, 6144),
    (6144, 2048),
    (2048, 6144),
]:
    packed = torch.randint(
        -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
    )
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    su = torch.ones(k, device="cuda", dtype=torch.float16)
    sv = torch.ones(n, device="cuda", dtype=torch.float16)
    rows = torch.arange(n, device="cuda") if n % 128 == 0 else None
    official = LinearEXL3(
        None,
        k,
        n,
        trellis=packed,
        suh=su,
        svh=sv,
        mul1=torch.tensor(True, device="cuda"),
    )
    for splits in (1, 3, 8, 16, 32, 64, 96):
        if splits > k // 64:
            continue
        kwargs = dict(
            splits=splits,
            arvq=True,
            input_scales=su,
            rows=rows,
            scales=sv if rows is not None else None,
        )
        expected = project(x, packed, lut, **kwargs)
        actual = project(x, packed, lut, gemv=True, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if args.parity_only:
            continue
        row = dict(
            k=k,
            n=n,
            splits=splits,
            generic_us=timing(lambda: project(x, packed, lut, **kwargs)),
            gemv_us=timing(lambda: project(x, packed, lut, gemv=True, **kwargs)),
        )
        if rows is not None:
            row["official_us"] = timing(
                lambda: official.forward(
                    x, {"reconstruct": False}, out_dtype=torch.float32
                )
            )
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
if not args.parity_only:
    Path("local-results/sm120_gemv.json").write_text(json.dumps(report, indent=2))
print("GEMV parity passed", flush=True)
