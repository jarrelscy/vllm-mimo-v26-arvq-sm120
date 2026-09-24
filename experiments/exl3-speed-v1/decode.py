# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode the four pilot artifacts using the official EXL3 1.5.1 runtime.

Usage: python decode.py layer21_expert201.bin output.pt
Output: gate_proj, up_proj, down_proj tensors in their original ordering.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from exllamav3.modules.quant.exl3 import LinearEXL3


def load(path, device="cuda"):
    raw = Path(path).read_bytes()
    meta = json.loads(raw[:4096])
    arrays = {}
    for spec in meta["arrays"]:
        blob = raw[spec["offset"] : spec["offset"] + spec["bytes"]]
        if spec["packed"]:
            a = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[
                : int(np.prod(spec["shape"]))
            ]
        else:
            a = np.frombuffer(blob, dtype=np.dtype(spec["dtype"]))
        arrays[spec["name"]] = torch.from_numpy(a.copy().reshape(spec["shape"])).to(
            device
        )
    assert arrays["keep"].shape == (2048,) and arrays["keep"].bool().all()
    values = [
        {
            "shape": shape,
            **{k.split(".")[1]: v for k, v in arrays.items() if k.startswith(f"{i}.")},
        }
        for i, shape in enumerate(meta["projections"])
    ]
    return values


@torch.no_grad()
def decode(path, device="cuda"):
    values = load(path, device)
    weights = {}
    for j, name in enumerate(("gate_proj", "up_proj", "down_proj")):
        pieces = [v for v in values if int(v["projection"]) == j]
        decoded = []
        for v in pieces:
            tensors = {
                k: t
                for k, t in v.items()
                if isinstance(t, torch.Tensor)
                and k not in ("projection", "blocks", "full_rows")
            }
            decoded.append(
                LinearEXL3(None, v["shape"][1], v["shape"][0], **tensors)
                .get_weight_tensor()
                .T.float()
            )
        if "blocks" in pieces[0]:
            weight = torch.empty(
                int(pieces[0]["full_rows"]), pieces[0]["shape"][1], device=device
            )
            for v, w in zip(pieces, decoded):
                rows = (
                    v["blocks"].long()[:, None] * 128 + torch.arange(128, device=device)
                ).flatten()
                weight[rows] = w
        else:
            weight = torch.cat(decoded, dim=0)
        weights[name] = weight
    return weights


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("output")
    args = ap.parse_args()
    torch.save({k: v.cpu() for k, v in decode(args.artifact).items()}, args.output)
