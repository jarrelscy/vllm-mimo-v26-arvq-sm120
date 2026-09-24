# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Synthetic MTP-shaped, pre-grouped routed MLP benchmark; no attention or TP comm."""

import argparse
import gc
import json
from pathlib import Path

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from native_fp4 import project
from sm120_microbench import timing
from trellis_gemm import make_pair_lut

parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, choices=[1, 4], default=4)
parser.add_argument("--parity-only", action="store_true")
args = parser.parse_args()
torch.manual_seed(9024)
torch.set_num_threads(4)
hidden, intermediate = 6144, 2048 // args.tp
lut = make_pair_lut("cuda")
report = {
    "tp_shard": args.tp,
    "gpu": torch.cuda.get_device_name(),
    "scope": (
        "Rate-2 synthetic cold experts, FP16 inputs, warm graphs, "
        "pre-grouped routing; gather/MLP/scatter included; router, attention, "
        "TP communication, drafting and verification logic excluded"
    ),
    "tuning": [],
    "cases": [],
}
path = Path(f"local-results/mtp_moe_tp{args.tp}.json")


def layer(k, n):
    packed = torch.randint(
        -32768, 32767, (k // 16, n // 16, 32), device="cuda", dtype=torch.int16
    )
    su = torch.ones(k, device="cuda", dtype=torch.float16)
    sv = torch.ones(n, device="cuda", dtype=torch.float16)
    official = LinearEXL3(
        None,
        k,
        n,
        trellis=packed,
        suh=su,
        svh=sv,
        mul1=torch.tensor(True, device="cuda"),
    )
    return packed, su, sv, torch.arange(n, device="cuda"), official


def native(x, entry, splits, generic=False):
    p, su, sv, rows, _ = entry
    return project(
        x,
        p,
        lut,
        splits=splits,
        rows=rows,
        scales=sv,
        input_scales=su,
        arvq=True,
        gemv=not generic,
    )


settings = {}
for label, k, n in [
    ("gate_up", hidden, 2 * intermediate),
    ("down", intermediate, hidden),
]:
    entry = layer(k, n)
    for m in [1, 2, 4]:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.01
        candidates = []
        for s in [4, 8, 16, 24, 32, 48, 64, 96]:
            if s > k // 64:
                continue
            value = 0 if args.parity_only else timing(lambda: native(x, entry, s))
            candidates.append((value, s))
        _, best = min(candidates)
        direct = (
            0
            if args.parity_only
            else timing(
                lambda: entry[-1].forward(
                    x, {"reconstruct": False}, out_dtype=torch.float32
                )
            )
        )
        recon = (
            1
            if args.parity_only
            else timing(
                lambda: entry[-1].forward(
                    x, {"reconstruct": True}, out_dtype=torch.float32
                )
            )
        )
        settings[label, m] = (best, recon < direct)
        report["tuning"].append(
            dict(
                projection=label,
                k=k,
                n=n,
                m=m,
                splits=best,
                native_us=min(candidates)[0],
                official_us=min(direct, recon),
                official_reconstruct=recon < direct,
            )
        )
    entry, x = None, None


def routes(name):
    if name == "draft_1":
        return [list(range(8))]
    if name == "verify_shared":
        return [list(range(8)) for _ in range(4)]
    if name == "verify_pairs":
        return [list(range(8 * (t // 2), 8 * (t // 2) + 8)) for t in range(4)]
    if name == "verify_disjoint":
        return [list(range(t * 8, (t + 1) * 8)) for t in range(4)]
    return [
        list(range(4)) + [4 + 2 * (t // 2), 5 + 2 * (t // 2)] + [8 + 2 * t, 9 + 2 * t]
        for t in range(4)
    ]


for scenario in [
    "draft_1",
    "verify_shared",
    "verify_pairs",
    "verify_mixed",
    "verify_disjoint",
]:
    route = routes(scenario)
    m = len(route)
    x = torch.randn(m, hidden, device="cuda", dtype=torch.float16) * 0.01
    rw = torch.rand(m, 8, device="cuda")
    rw /= rw.sum(1, keepdim=True)
    ids = sorted({e for row in route for e in row})
    experts = []
    histogram = {}
    for expert in ids:
        selections = [
            (t, row.index(expert)) for t, row in enumerate(route) if expert in row
        ]
        toks = torch.tensor(
            [t for t, _ in selections], device="cuda", dtype=torch.int64
        )
        weights = rw[
            toks, torch.tensor([j for _, j in selections], device="cuda")
        ].view(-1, 1)
        histogram[len(selections)] = histogram.get(len(selections), 0) + 1
        experts.append(
            (
                toks,
                weights,
                layer(hidden, 2 * intermediate),
                layer(intermediate, hidden),
            )
        )

    def run(mode):
        output = torch.zeros((m, hidden), device="cuda", dtype=torch.float32)
        for toks, weight, w13, w2 in experts:
            inp = x.index_select(0, toks)
            size = toks.numel()
            split13, recon13 = settings["gate_up", size]
            split2, recon2 = settings["down", size]
            if mode == "official":
                gu = w13[-1].forward(
                    inp, {"reconstruct": recon13}, out_dtype=torch.float32
                )
            else:
                gu = native(inp, w13, split13, generic=mode == "generic")
            gate, up = gu.chunk(2, dim=-1)
            h = (torch.nn.functional.silu(gate) * up).half()
            if mode == "official":
                down = w2[-1].forward(
                    h, {"reconstruct": recon2}, out_dtype=torch.float32
                )
            else:
                down = native(h, w2, split2, generic=mode == "generic")
            output.index_add_(0, toks, down * weight)
        return output

    reference = run("generic")
    actual = run("native")
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    comparisons = []
    if not args.parity_only:
        for order in [("native", "official"), ("official", "native")]:
            comparisons.append({mode: timing(lambda: run(mode)) for mode in order})
    row = dict(
        scenario=scenario,
        tokens=m,
        active_experts=len(ids),
        expert_token_histogram=histogram,
        parity="bit exact against generic FP4",
        samples=comparisons,
    )
    report["cases"].append(row)
    print(json.dumps(row), flush=True)
    if not args.parity_only:
        path.write_text(json.dumps(report, indent=2))
    experts, reference, actual, x = [], None, None, None
    gc.collect()
print("MTP-shaped MoE parity passed", flush=True)
