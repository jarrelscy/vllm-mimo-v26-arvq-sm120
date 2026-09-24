# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
import json
from pathlib import Path

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from native_fp4 import project
from sm120_microbench import timing
from trellis_gemm import make_pair_lut

torch.set_num_threads(4)
results = []
for seed in [71, 97, 123]:
    torch.manual_seed(seed)
    for k, n, s in [(6144, 2048, 32), (512, 6144, 8)]:
        p = torch.randint(
            -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
        )
        lut = make_pair_lut("cuda")
        su = torch.randn(k, device="cuda", dtype=torch.float16)
        sv = torch.randn(n, device="cuda", dtype=torch.float16)
        rows = torch.arange(n, device="cuda")
        x = torch.randn(2, k, device="cuda", dtype=torch.float16)
        o = LinearEXL3(
            None,
            k,
            n,
            trellis=p,
            suh=su,
            svh=sv,
            mul1=torch.tensor(True, device="cuda"),
        )
        kw = dict(splits=s, rows=rows, scales=sv, input_scales=su, arvq=True)
        torch.testing.assert_close(
            project(x, p, lut, gemv=True, **kw),
            project(x, p, lut, **kw),
            rtol=0,
            atol=0,
        )
        funcs = {
            "fp4": lambda: project(x, p, lut, gemv=True, **kw),
            "official_direct": lambda: o.forward(
                x, {"reconstruct": False}, out_dtype=torch.float32
            ),
            "official_reconstruct": lambda: o.forward(
                x, {"reconstruct": True}, out_dtype=torch.float32
            ),
        }
        for order in [list(funcs), list(reversed(funcs))]:
            r = dict(seed=seed, k=k, n=n, m=2, order=order)
            for name in order:
                r[name] = timing(funcs[name])
            results.append(r)
            print(json.dumps(r), flush=True)
Path("local-results/confirmed_two.json").write_text(json.dumps(results, indent=2))
