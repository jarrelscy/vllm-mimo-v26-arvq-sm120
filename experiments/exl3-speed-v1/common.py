# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paths and source/capture helpers for the four MiMo pilots."""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/mimo_arvq"))
from source import Source  # noqa: E402,F401

ARTIFACT_ROOT = Path(
    os.environ.get("MIMO_ARTIFACT_ROOT", "/data/jarrel/mimo-known-baselines-v1")
)
WORK_ROOT = Path(os.environ.get("MIMO_WORK_ROOT", "/data/jarrel/mimo-v26-arvq-hot5"))
RESULT_ROOT = Path(
    os.environ.get("MIMO_RESULT_ROOT", str(Path(__file__).parent / "local-results"))
)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
CANDIDATES = [
    (21, 137, "block_allocation_layer21_expert137_lower1/joint", "blocks"),
    (21, 32, "exl3_allocation_layer21_expert32/joint", "contiguous"),
    (21, 201, "two_sided_layer21_expert201_sigma0.1_out1/joint", "contiguous"),
    (69, 19, "block_allocation_layer69_expert19_lower1/joint", "blocks"),
]


def load_data(work, layer, expert):
    records = {s: {"x": [], "p": []} for s in (0, 1, 2)}
    for rank in range(8):
        part = torch.load(
            work / f"capture{layer}/rank{rank}.pt", mmap=True, weights_only=True
        )
        route = ((part["topk_ids"] == expert) * part["topk_weights"]).sum(1)
        for s in records:
            mask = (part["pv_split"] == s) & (route != 0)
            records[s]["x"].append(part["x"][mask])
            records[s]["p"].append(route[mask])
    return {
        s: {k: torch.cat(v).float().cuda() for k, v in rec.items()}
        for s, rec in records.items()
    }


def forward(x, weights):
    g, u, d = weights
    return F.linear(
        F.silu(F.linear(x.bfloat16(), g.bfloat16()))
        * F.linear(x.bfloat16(), u.bfloat16()),
        d.bfloat16(),
    ).float()
