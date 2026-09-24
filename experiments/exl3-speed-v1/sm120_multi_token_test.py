# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
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
rs = []
for k, n in [(6144, 512), (512, 6144), (6144, 2048), (2048, 6144)]:
    p = torch.randint(
        -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
    )
    lut = make_pair_lut("cuda")
    su = torch.ones(k, device="cuda", dtype=torch.float16)
    sv = torch.ones(n, device="cuda", dtype=torch.float16)
    rows = torch.arange(n, device="cuda")
    o = LinearEXL3(
        None, k, n, trellis=p, suh=su, svh=sv, mul1=torch.tensor(True, device="cuda")
    )
    for m in [1, 2, 3, 4, 5, 8]:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        official = (
            None
            if args.parity_only
            else timing(
                lambda: o.forward(x, {"reconstruct": False}, out_dtype=torch.float32)
            )
        )
        reconstruction = (
            None
            if args.parity_only
            else timing(
                lambda: o.forward(x, {"reconstruct": True}, out_dtype=torch.float32)
            )
        )
        for s in [8, 16, 32, 48, 64, 96]:
            if s > k // 64:
                continue
            kw = dict(splits=s, rows=rows, scales=sv, input_scales=su, arvq=True)
            a = project(x, p, lut, **kw)
            b = project(x, p, lut, gemv=True, **kw)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            if args.parity_only:
                continue
            r = dict(
                k=k,
                n=n,
                m=m,
                s=s,
                official=official,
                reconstruct=reconstruction,
                fp4=timing(lambda: project(x, p, lut, gemv=True, **kw)),
            )
            rs.append(r)
            print(json.dumps(r), flush=True)
if not args.parity_only:
    Path("local-results/multi_token.json").write_text(json.dumps(rs, indent=2))
print("Multi-token parity passed", flush=True)
