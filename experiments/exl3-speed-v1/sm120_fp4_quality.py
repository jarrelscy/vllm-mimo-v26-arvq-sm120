# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Emulate FP4 x FP4 MMA operands; quality only, no hardware timing."""

import argparse
import json

import torch
import torch.nn.functional as F
from common import (
    ARTIFACT_ROOT,
    CANDIDATES,
    RESULT_ROOT,
    WORK_ROOT,
    Source,
    forward,
    load_data,
)
from decode import load
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3


def nearest(x, levels):
    idx = torch.searchsorted(levels, x.contiguous()).clamp(1, len(levels) - 1)
    lo, hi = levels[idx - 1], levels[idx]
    return torch.where((x - lo).abs() <= (x - hi).abs(), lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--group", type=int, choices=(16, 32), default=32)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer, expert, subdir, _ = CANDIDATES[args.index]
    values = load(ARTIFACT_ROOT / subdir / "selected.bin")
    work = WORK_ROOT
    data = load_data(work, layer, expert)
    source = Source(work / "source")
    native = [
        source.expert(layer, expert, p, "cuda")
        for p in ("gate_proj", "up_proj", "down_proj")
    ]
    levels = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda"
    )
    pair = (levels[:, None] + levels[None, :] / 16).flatten().unique().sort().values
    groups = []
    for j in range(3):
        parts = []
        offset = 0
        for v in values:
            if int(v["projection"]) != j:
                continue
            tensors = {
                k: t
                for k, t in v.items()
                if isinstance(t, torch.Tensor)
                and k not in ("projection", "blocks", "full_rows")
            }
            obj = LinearEXL3(None, v["shape"][1], v["shape"][0], **tensors)
            w = nearest(obj.get_inner_weight_tensor().float(), pair).half()
            rows = (
                (
                    v["blocks"].long()[:, None] * 128 + torch.arange(128, device="cuda")
                ).flatten()
                if "blocks" in v
                else torch.arange(offset, offset + v["shape"][0], device="cuda")
            )
            offset += v["shape"][0]
            parts.append((obj, w, rows))
        groups.append((parts, offset))

    def activation(x, planes):
        original = x.float().reshape(len(x), -1, args.group)
        remaining = original
        reconstructed = torch.zeros_like(original)
        for _ in range(planes):
            peak = remaining.abs().amax(-1, keepdim=True)
            scale = torch.exp2(torch.ceil(torch.log2((peak / 6).clamp_min(2**-24))))
            q = nearest(remaining / scale, levels) * scale
            reconstructed = reconstructed + q
            remaining = original - reconstructed
        return reconstructed.reshape_as(x).half()

    def project(x, j, planes):
        parts, n = groups[j]
        result = torch.empty(len(x), n, device="cuda")
        for obj, w, rows in parts:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, obj.suh, None, 1.0)
            y = activation(xh, planes).float() @ w.float()
            ext.had_r_128(y, y, None, obj.svh, 1.0)
            result[:, rows] = y
        return result

    def run(x, planes):
        g, u = project(x, 0, planes), project(x, 1, planes)
        return project((F.silu(g) * u).half(), 2, planes)

    report = {
        "layer": layer,
        "expert": expert,
        "group_size": args.group,
        "scope": (
            "FP4 operand numerical emulation on B200; no SM120 "
            "execution or speed claim; two weight planes, independently "
            "scaled activation planes"
        ),
        "candidates": [],
    }
    with torch.no_grad():
        targets = {s: forward(data[s]["x"], native) for s in (1, 2)}
        for planes in (1, 2):
            row = {
                "activation_planes": planes,
                "weight_planes": 2,
                "mma_terms": planes * 2,
            }
            for s, name in ((1, "validation"), (2, "audit")):
                x, p = data[s]["x"], data[s]["p"][:, None]
                pred = torch.cat(
                    [run(chunk.half().contiguous(), planes) for chunk in x.split(64)]
                )
                row[name] = float(
                    ((pred - targets[s]) * p).double().norm()
                    / (targets[s] * p).double().norm()
                )
            report["candidates"].append(row)
            print(row, flush=True)
    out = RESULT_ROOT / f"sm120_fp4_quality_g{args.group}_l{layer}_e{expert}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
