# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent integer decoder oracle and emitted instruction inspection."""

import json
from pathlib import Path

import numpy as np
import torch
from trellis_gemm import gemm_fp4, make_lut, make_pair_lut, multiply

torch.manual_seed(91426)
reports = []
for rate in (1.5, 2, 2.5):
    k = n = 256
    packed = torch.randint(
        -32768, 32767, (k // 16, n // 16, int(16 * rate)), dtype=torch.int16
    )
    words = packed.numpy().view(np.uint32)
    exact = make_lut("cuda")
    pair = make_pair_lut("cuda")
    cpu_lut = exact.cpu().numpy()
    pair_lut = pair.cpu().numpy()
    levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    expected = np.empty((k, n), dtype=np.float32)
    approx = np.empty_like(expected)
    for r in range(k):
        for c in range(n):
            rr, cc = r % 16, c % 16
            lane = (cc % 8) * 4 + (rr % 8) // 2
            element = rr % 2 + 2 * (rr // 8) + 4 * (cc // 8)
            pos = lane * 8 + element
            bits = int(256 * rate)
            end = (pos + 1) * int(rate) + ((pos + 1) // 2 if rate % 1 else 0) + bits
            block = words[r // 16, c // 16]
            # Assemble the circular bitstream one bit at a time, MSB first.
            state = 0
            for bit in range(end - 16, end):
                at = bit % bits
                state = (state << 1) | ((int(block[at // 32]) >> (31 - at % 32)) & 1)
            value = (state * 0x83DCD12D) & 0xFFFFFFFF
            index = sum((value >> shift) & 255 for shift in (0, 8, 16, 24))
            expected[r, c] = cpu_lut[index]
            code = int(pair_lut[index])
            approx[r, c] = levels[code & 15] + levels[code >> 4] / 16
    packed = packed.cuda()
    identity = torch.eye(k, device="cuda", dtype=torch.float16)
    actual = multiply(identity, packed, exact, splits=2)
    torch.testing.assert_close(
        actual, torch.tensor(expected, device="cuda"), rtol=0, atol=0
    )
    actual_pair = multiply(identity, packed, pair, splits=2, fp4=True, residual=True)
    torch.testing.assert_close(
        actual_pair, torch.tensor(approx, device="cuda"), rtol=0, atol=0
    )
    x = torch.randn(4, k, device="cuda", dtype=torch.float16)
    pred = multiply(x, packed, pair, splits=2, fp4=True, residual=True)
    ref = x.float() @ torch.tensor(approx, device="cuda")
    reports.append(
        {
            "rate": rate,
            "exact_decode": True,
            "pair_decode": True,
            "activation_gemm_relative_l2": ((pred - ref).norm() / ref.norm()).item(),
            "max_abs": (pred - ref).abs().max().item(),
        }
    )

x = torch.randn(1, 256, device="cuda", dtype=torch.float16)
y = torch.empty((2, 1, 256), device="cuda")
compiled = gemm_fp4[(4, 1, 2)](
    x,
    packed.view(torch.int32),
    pair,
    y,
    1,
    256,
    256,
    2,
    1,
    2,
    True,
    BM=16,
    BN=64,
    num_warps=4,
)
Path("local-results/sm120_fp4.ptx").write_text(compiled.asm["ptx"])
Path("local-results/sm120_fp4.cubin").write_bytes(compiled.asm["cubin"])
Path("local-results/sm120_verify.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
print(
    "MMA instructions:",
    sorted(
        set(line.strip() for line in compiled.asm["ptx"].splitlines() if "mma." in line)
    ),
)
