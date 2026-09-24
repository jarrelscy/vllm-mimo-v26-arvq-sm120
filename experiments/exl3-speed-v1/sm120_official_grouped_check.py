# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the official fused adapter against reconstructed EXL3 weights."""

import torch
from official_grouped import OfficialGrouped

E, H, intermediate = 8, 6144, 512
parts = []
for k, n in [(H, intermediate), (H, intermediate), (intermediate, H)]:
    parts.append(
        (
            torch.randint(
                -32768,
                32767,
                (E, k // 16, n // 16, 32),
                device="cuda",
                dtype=torch.int16,
            ),
            torch.randn(E, k, device="cuda", dtype=torch.float16),
            torch.randn(E, n, device="cuda", dtype=torch.float16),
        )
    )
o = OfficialGrouped(*parts)
for m in [1, 2, 3, 4]:
    x = torch.randn(m, H, device="cuda", dtype=torch.float16) * 0.01
    sel = torch.stack([torch.randperm(8, device="cuda") for _ in range(m)])
    weights = torch.rand(m, 8, device="cuda").softmax(-1).half()
    ref = torch.zeros(m, H, device="cuda")
    for t in range(m):
        for j in range(8):
            e = sel[t, j].item()
            gu = [
                o.layers[q][e].forward(
                    x[t : t + 1], {"reconstruct": True}, out_dtype=torch.float32
                )
                for q in [0, 1]
            ]
            h = (torch.nn.functional.silu(gu[0]) * gu[1]).half()
            y = o.layers[2][e].forward(
                h, {"reconstruct": True}, out_dtype=torch.float32
            )
            ref[t : t + 1] += y * weights[t, j].float()
    a = o(x, sel, weights).clone()
    b = o(x, sel, weights).clone()
    rel = ((a - ref).norm() / ref.norm()).item()
    assert rel < 0.003, (m, rel)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    print("official grouped vs reconstruct", m, rel, flush=True)
