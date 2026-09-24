# ruff: noqa: B023
# Benchmark callbacks execute synchronously within each loop iteration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare official EXL3 and custom paths with identical synthetic weights."""

import json
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from sm120_microbench import timing
from trellis_gemm import make_lut, make_pair_lut, multiply_scatter

torch.manual_seed(123)
torch.set_num_threads(4)
lut = make_lut("cuda")
pair = make_pair_lut("cuda")
report = {
    "gpu": torch.cuda.get_device_name(),
    "scope": (
        "Synthetic rate-2 single projections; official scaled Hadamard; "
        "no routing or TP communication"
    ),
    "rows": [],
}
for k, n in [(6144, 512), (512, 6144), (6144, 2048), (2048, 6144)]:
    packed = torch.randint(
        -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
    )
    su, sv = (
        torch.ones(k, device="cuda", dtype=torch.float16),
        torch.ones(n, device="cuda", dtype=torch.float16),
    )
    obj = LinearEXL3(
        None,
        k,
        n,
        trellis=packed,
        suh=su,
        svh=sv,
        mul1=torch.tensor(True, device="cuda"),
    )
    obj.config.infer_params.no_reconstruct = True
    rows = torch.arange(n, device="cuda")
    for m in (1, 4, 16, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        xh = torch.empty_like(x)
        y = torch.empty((m, n), device="cuda", dtype=torch.float32)

        def had():
            ext.had_r_128(x, xh, su, None, 1.0)

        def custom(fp4=False):
            had()
            multiply_scatter(
                xh,
                packed,
                pair if fp4 else lut,
                y,
                rows,
                sv,
                splits=8,
                fp4=fp4,
                residual=fp4,
            )
            return y

        reference = obj.forward(x, {"reconstruct": True}, out_dtype=torch.float32)
        actual = custom().clone()
        discrepancy = ((actual - reference).norm() / reference.norm()).item()
        assert discrepancy < 0.005, discrepancy
        row = dict(
            m=m,
            k=k,
            n=n,
            custom_fp16_relative_l2=discrepancy,
            official_hadamard_us=timing(had),
            official_direct_us=timing(
                lambda: obj.forward(x, {"reconstruct": False}, out_dtype=torch.float32)
            ),
            official_reconstruct_us=timing(
                lambda: obj.forward(x, {"reconstruct": True}, out_dtype=torch.float32)
            ),
            custom_fp16_us=timing(custom),
            custom_fp4_us=timing(lambda: custom(True)),
        )
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
        Path("local-results/sm120_official_bench.json").write_text(
            json.dumps(report, indent=2)
        )
