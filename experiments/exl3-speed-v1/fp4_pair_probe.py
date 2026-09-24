# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quality-only probe: map decoded trellis symbols to two FP4 planes."""

import argparse
import json

import torch
from common import (
    ARTIFACT_ROOT,
    CANDIDATES,
    RESULT_ROOT,
    WORK_ROOT,
    Source,
    forward,
    load_data,
)
from decode import decode, load
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.exl3_lib.quantize import preapply_had_l, preapply_had_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer, expert, subdir, codec = CANDIDATES[args.index]
    root = ARTIFACT_ROOT
    work = WORK_ROOT
    path = root / subdir / "selected.bin"
    values = load(path)
    list(decode(path).values())
    source = Source(work / "source")
    data = load_data(work, layer, expert)
    native = [
        source.expert(layer, expert, p, "cuda")
        for p in ("gate_proj", "up_proj", "down_proj")
    ]
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    targets = {s: forward(data[s]["x"], native) for s in (1, 2)}
    report = {
        "layer": layer,
        "expert": expert,
        "scope": (
            "quality-only; no speed claim; planes generated from "
            "trellis symbols, not stored as extra weight arrays"
        ),
        "candidates": [],
    }
    with torch.no_grad():
        inner = []
        for v in values:
            tensors = {
                k: t
                for k, t in v.items()
                if isinstance(t, torch.Tensor)
                and k not in ("projection", "blocks", "full_rows")
            }
            inner.append(
                LinearEXL3(None, v["shape"][1], v["shape"][0], **tensors)
                .get_inner_weight_tensor()
                .float()
            )
        for shift in (2, 3, 4, 5):
            for scale in (0.5, 1.0):
                candidates = (
                    (scale * (levels[:, None] + levels[None, :] / 2**shift))
                    .flatten()
                    .sort()
                    .values.unique()
                )
                pieces = []
                errnum = 0.0
                errden = 0.0
                for v, w in zip(values, inner):
                    right = torch.searchsorted(candidates, w.flatten()).clamp(
                        max=len(candidates) - 1
                    )
                    left = (right - 1).clamp_min(0)
                    chosen = torch.where(
                        (w.flatten() - candidates[left]).abs()
                        <= (w.flatten() - candidates[right]).abs(),
                        left,
                        right,
                    )
                    q = candidates[chosen].reshape_as(w)
                    errnum += float((q - w).square().sum())
                    errden += float(w.square().sum())
                    # Match official reconstruction rounding boundaries.
                    q = preapply_had_l(q.half(), 128)
                    q *= v["suh"][:, None]
                    q = preapply_had_r(q, 128)
                    q *= v["svh"][None, :]
                    pieces.append(q.T.float())
                ws = []
                for j in range(3):
                    selected = [
                        (v, w)
                        for v, w in zip(values, pieces)
                        if int(v["projection"]) == j
                    ]
                    if "blocks" in selected[0][0]:
                        v = selected[0][0]
                        result = torch.empty(
                            int(v["full_rows"]), v["shape"][1], device="cuda"
                        )
                        for v, w in selected:
                            rows = (
                                v["blocks"].long()[:, None] * 128
                                + torch.arange(128, device="cuda")
                            ).flatten()
                            result[rows] = w
                    else:
                        result = torch.cat([w for _, w in selected])
                    ws.append(result)
                row = {
                    "residual_shift": shift,
                    "scale": scale,
                    "symbol_relative_rms": (errnum / errden) ** 0.5,
                }
                for s, name in ((1, "validation"), (2, "audit")):
                    p = data[s]["p"][:, None]
                    pred = forward(data[s]["x"], ws)
                    row[name] = float(
                        ((pred - targets[s]) * p).double().norm()
                        / (targets[s] * p).double().norm()
                    )
                report["candidates"].append(row)
                print(row, flush=True)
    out = RESULT_ROOT
    out.mkdir(exist_ok=True)
    (out / f"fp4_pair_layer{layer}_expert{expert}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
