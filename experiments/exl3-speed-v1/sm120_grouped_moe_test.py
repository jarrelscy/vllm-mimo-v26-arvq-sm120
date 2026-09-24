# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Native grouped FP4 versus official two-launch small-batch EXL3 MoE."""

import argparse
import json
from pathlib import Path

import torch
from grouped_fp4 import GroupedMoE
from native_fp4 import project
from official_grouped import OfficialGrouped
from sm120_microbench import timing
from trellis_gemm import make_pair_lut

parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, choices=[1, 4], default=4)
parser.add_argument("--parity-only", action="store_true")
args = parser.parse_args()
torch.manual_seed(71043)
torch.set_num_threads(4)
E, H, intermediate, topk = 32, 6144, 2048 // args.tp, 8
lut = make_pair_lut("cuda")
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
            torch.ones(E, k, device="cuda", dtype=torch.float16),
            torch.ones(E, n, device="cuda", dtype=torch.float16),
        )
    )
official = OfficialGrouped(*parts)
report = {"tp": args.tp, "gpu": torch.cuda.get_device_name(), "rows": []}
for m in [1, 2, 3, 4]:
    x = torch.randn(m, H, device="cuda", dtype=torch.float16) * 0.01
    weights = torch.rand(m, topk, device="cuda", dtype=torch.float16)
    weights = (weights.float() / weights.float().sum(1, keepdim=True)).half()
    for pattern in ["shared", "pairs", "disjoint"]:
        ids = [
            list(
                range(
                    8
                    * (
                        0
                        if pattern == "shared"
                        else (t // 2 if pattern == "pairs" else t)
                    ),
                    8
                    * (
                        0
                        if pattern == "shared"
                        else (t // 2 if pattern == "pairs" else t)
                    )
                    + 8,
                )
            )
            for t in range(m)
        ]
        selected = torch.tensor(ids, device="cuda", dtype=torch.int64)
        ref = torch.zeros(m, H, device="cuda")
        for t in range(m):
            for j, e in enumerate(ids[t]):
                outputs = []
                for p, su, sv in parts[:2]:
                    outputs.append(
                        project(
                            x[t : t + 1],
                            p[e],
                            lut,
                            splits=min(8, H // 64),
                            rows=torch.arange(intermediate, device="cuda"),
                            scales=sv[e],
                            input_scales=su[e],
                            arvq=True,
                            gemv=True,
                        )
                    )
                activation = (torch.nn.functional.silu(outputs[0]) * outputs[1]).half()
                p, su, sv = parts[2]
                y = project(
                    activation,
                    p[e],
                    lut,
                    splits=min(4, intermediate // 64),
                    rows=torch.arange(H, device="cuda"),
                    scales=sv[e],
                    input_scales=su[e],
                    arvq=True,
                    gemv=True,
                )
                ref[t : t + 1] += y * weights[t, j].float()
        official_time = (
            0 if args.parity_only else timing(lambda: official(x, selected, weights))
        )
        for sg in [1, 2, 4, 8]:
            for sd in [1, 2, 4]:
                if sd > intermediate // 64:
                    continue
                native = GroupedMoE(*parts, lut, m, splits_gu=sg, splits_down=sd)
                out = native(x, selected, weights)
                rel = ((out - ref).norm() / ref.norm()).item()
                assert torch.isfinite(out).all() and rel < 0.001, (
                    m,
                    pattern,
                    sg,
                    sd,
                    rel,
                )
                row = dict(
                    m=m,
                    pattern=pattern,
                    sg=sg,
                    sd=sd,
                    relative_l2=rel,
                    official_us=official_time,
                )
                if not args.parity_only:
                    row["native_us"] = timing(lambda: native(x, selected, weights))
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                if not args.parity_only:
                    Path(f"local-results/grouped_moe_tp{args.tp}.json").write_text(
                        json.dumps(report, indent=2)
                    )
print("Grouped MoE parity passed", flush=True)
