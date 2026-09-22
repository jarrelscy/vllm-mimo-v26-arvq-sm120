# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanity-check the complete native-weight capture trajectory on held-out text."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from reference import Layer, rms
from source import Source


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.accelerator.set_device_index(torch.device(args.device).index)
    source = Source(args.work / "source")
    raw = np.load(args.work / "corpus/final_test.npy").reshape(-1, 1024)
    # One complete sequence from each of the six fixed category allocations.
    tokens = torch.tensor(
        raw[[0, 3, 6, 10, 12, 14]].astype(np.int64), device=args.device
    )
    embedding = source.tensor("model.embed_tokens.weight", args.device).bfloat16()
    state = F.embedding(tokens, embedding)
    del embedding
    for number in range(70):
        layer = Layer(source, number, args.device)
        states = []
        for chunk in state.split(2):
            if number == 0:
                states.append(layer.dense_forward(chunk))
            else:
                residual, x = layer.front(chunk)
                flat = x.flatten(0, 1)
                ids, gates = layer.route(flat)
                output = layer.native_moe(flat, ids, gates)
                states.append(residual + output.reshape_as(residual).to(residual.dtype))
                del residual, x, flat, output, ids, gates
        state = torch.cat(states)
        del states
        if not torch.isfinite(state).all():
            raise ValueError(f"Nonfinite native trajectory at layer {number}")
        del layer
        print("REFERENCE", number, flush=True)
    norm = source.tensor("model.norm.weight", args.device).bfloat16()
    head = source.tensor("lm_head.weight", args.device).bfloat16()
    hidden = rms(state, norm)[:, :-1].flatten(0, 1)
    labels = tokens[:, 1:].flatten()
    total = 0.0
    for offset in range(0, len(labels), 128):
        sl = slice(offset, offset + 128)
        logits = F.linear(hidden[sl], head).float()
        total += float(F.cross_entropy(logits, labels[sl], reduction="sum"))
    nll = total / len(labels)
    if not math.isfinite(nll):
        raise ValueError("Nonfinite reference likelihood")
    report = {
        "complete": True,
        "tokens_scored": len(labels),
        "nll": nll,
        "perplexity": math.exp(nll),
        "scope": "Native-weight capture emulation sanity check, not serving parity",
    }
    (args.work / "reference_likelihood.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
