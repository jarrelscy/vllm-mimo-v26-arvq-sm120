# ruff: noqa: B023
# Benchmark callbacks execute synchronously within each loop iteration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic SM120 decomposition; no fitted-checkpoint quality claims."""

import argparse
import json
from pathlib import Path

import torch
from trellis_gemm import (
    make_lut,
    make_pair_lut,
    multiply,
    multiply_scatter,
    reduce,
    reduce_had_scatter,
)

from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr


@tr.jit
def had128(X, Y, SIZE: tl.constexpr):
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    lane = tl.arange(0, 128)
    v = tl.load(X + i, i < SIZE, 0).to(tl.float32)
    for stage in tl.static_range(7):
        other = tl.gather(v, lane ^ (1 << stage), 0)
        v = tl.where((lane & (1 << stage)) == 0, v + other, other - v)
    tl.store(Y + i, v * 0.08838834764831845, i < SIZE)


def timing(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = (
        torch.Event(enable_timing=True),
        torch.Event(enable_timing=True),
    )
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 10)
    return sorted(samples)[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp4", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.set_num_threads(4)
    lut = make_pair_lut("cuda") if args.fp4 else make_lut("cuda")
    # Independently verify normalized block Hadamard against a dense matrix.
    h = torch.ones((1, 1), device="cuda")
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= 128**0.5
    test = torch.randn(4, 128, device="cuda")
    out = torch.empty_like(test)
    had128[(4,)](test, out, test.numel())
    torch.testing.assert_close(out, test @ h, atol=2e-6, rtol=2e-5)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "fp4": args.fp4,
        "scope": (
            "Synthetic one-projection timings, warm graphs; no routing/TP "
            "communication; Hadamard excludes fitted scales"
        ),
        "rows": [],
    }
    for k, n in [(6144, 512), (512, 6144), (6144, 2048), (2048, 6144)]:
        packed = torch.randint(
            -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
        )
        for m in (1, 4, 16, 64):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            xh = torch.empty_like(x)
            y = torch.empty((m, n), device="cuda", dtype=torch.float32)
            rows = torch.arange(n, device="cuda")
            scale = torch.ones(n, device="cuda")
            scratch = torch.randn(8, m, n, device="cuda")

            def ih():
                had128[(m * k // 128,)](x, xh, m * k)

            def oh():
                had128[(m * n // 128,)](y, y, m * n)

            def fused_ep():
                reduce_had_scatter[(m, n // 128)](
                    scratch, y, rows, scale, m, n, n, 8, num_warps=4
                )

            def separate_ep():
                reduce[(tr.cdiv(m * n, 256),)](scratch, y, m, n, 8, 256, num_warps=4)
                oh()

            def product():
                return multiply(
                    x, packed, lut, splits=8, fp4=args.fp4, residual=args.fp4
                )

            def complete():
                ih()
                multiply_scatter(
                    xh,
                    packed,
                    lut,
                    y,
                    rows,
                    scale,
                    splits=8,
                    fp4=args.fp4,
                    residual=args.fp4,
                )

            row = {
                "m": m,
                "k": k,
                "n": n,
                "input_hadamard_us": timing(ih),
                "output_hadamard_us": timing(oh),
                "separate_reduce_had_us": timing(separate_ep),
                "fused_reduce_had_scatter_us": timing(fused_ep),
                "decode_gemm_reduce_us": timing(product),
                "complete_projection_us": timing(complete),
            }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            Path(
                "local-results/sm120_microbench_"
                + ("fp4" if args.fp4 else "fp16")
                + ".json"
            ).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
