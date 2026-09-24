# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic trellis correctness test; no MiMo checkpoint is required."""

import argparse
import json

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from trellis_gemm import make_lut, multiply, multiply_scatter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(91426)
    device = args.device
    lut = make_lut(device)
    reports = []
    for rate in (1.5, 2, 2.5):
        k, n = 256, 256
        packed = torch.randint(
            -32768,
            32767,
            (k // 16, n // 16, int(16 * rate)),
            device=device,
            dtype=torch.int16,
        )
        suh = torch.ones(k, device=device, dtype=torch.float16)
        svh = torch.ones(n, device=device, dtype=torch.float16)
        obj = LinearEXL3(
            None,
            k,
            n,
            trellis=packed,
            suh=suh,
            svh=svh,
            mul1=torch.tensor(True, device=device),
        )
        expected = obj.get_inner_weight_tensor()
        identity = torch.eye(k, device=device, dtype=torch.float16)
        actual = multiply(identity, packed, lut, splits=2)
        torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
        for m in (1, 4, 16, 64):
            x = torch.randn(m, k, device=device, dtype=torch.float16)
            reference = x.float() @ expected.float()
            product = multiply(x, packed, lut, splits=2)
            torch.testing.assert_close(product, reference, rtol=2e-4, atol=2e-4)
            ext.had_r_128(reference, reference, None, svh, 1.0)
            rows = torch.cat(
                (
                    torch.arange(128, 256, device=device),
                    torch.arange(128, device=device),
                )
            )
            output = torch.empty(m, n, device=device, dtype=torch.float32)
            multiply_scatter(x, packed, lut, output, rows, svh, splits=2)
            torch.testing.assert_close(output[:, rows], reference, rtol=3e-4, atol=3e-4)
        reports.append({"rate": rate, "decoded_exact": True, "batches": [1, 4, 16, 64]})
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "passed": reports}))


if __name__ == "__main__":
    main()
