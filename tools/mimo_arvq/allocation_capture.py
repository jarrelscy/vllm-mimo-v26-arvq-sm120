# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small training-only native trajectory for global hot-allocation scoring."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "recipe"))
from arvq88.inputs import write
from reference import Layer
from source import Source


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work, parent = args.work, args.parent
    work.mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    dev = f"cuda:{rank}"
    if rank == 0:
        for name in ("source", "corpus"):
            if not (work / name).exists():
                (work / name).symlink_to(parent / name, target_is_directory=True)
        write(work / "config.json", json.loads((parent / "config.json").read_text()))
    dist.barrier()
    source = Source(work / "source")
    tokens = np.load(work / "corpus/train.npy", mmap_mode="r")
    sequences = np.linspace(0, len(tokens) // 1024 - 1, 64).astype(np.int64)
    local = sequences[rank::world]
    ids = torch.tensor(
        np.stack([tokens[s * 1024 : (s + 1) * 1024] for s in local]),
        device=dev,
        dtype=torch.long,
    )
    complete = sorted(work.glob("native_layer_*_complete.json"))
    if complete:
        start = json.loads(complete[-1].read_text())["layer"] + 1
        state = torch.load(
            work / f"native_layer_{start - 1:03d}_rank{rank}.pt",
            map_location=dev,
            weights_only=True,
        )
    else:
        start = 0
        embedding = source.tensor("model.embed_tokens.weight", dev).bfloat16()
        state = F.embedding(ids, embedding)
        del embedding
    for number in range(start, 70):
        layer = Layer(source, number, dev)
        states, captures = [], []
        for batch in state.split(2):
            if number == 0:
                states.append(layer.dense_forward(batch))
            else:
                residual, x = layer.front(batch)
                flat = x.flatten(0, 1)
                expert_ids, gates = layer.route(flat)
                native = layer.native_moe(flat, expert_ids, gates)
                captures.append(
                    {
                        "x": flat.cpu(),
                        "topk_ids": expert_ids.cpu(),
                        "topk_weights": gates.cpu(),
                    }
                )
                states.append(residual + native.reshape_as(residual).to(residual.dtype))
        state = torch.cat(states)
        if not torch.isfinite(state).all():
            raise ValueError(f"Nonfinite allocation trajectory at {number}")
        if number:
            data = {key: torch.cat([c[key] for c in captures]) for key in captures[0]}
            data["pv_split"] = torch.zeros(len(data["x"]), dtype=torch.long)
            folder = work / f"capture{number}"
            folder.mkdir(exist_ok=True)
            torch.save(data, folder / f"rank{rank}.pt")
            fit_folder = work / f"training_capture{number}"
            fit_folder.mkdir(exist_ok=True)
            torch.save(data, fit_folder / f"train_rank{rank}_00000.pt")
        torch.save(state.cpu(), work / f"native_layer_{number:03d}_rank{rank}.pt")
        del layer, states, captures
        dist.barrier()
        if rank == 0:
            write(work / f"native_layer_{number:03d}_complete.json", {"layer": number})
            if number:
                for old in work.glob(f"native_layer_{number - 1:03d}_rank*.pt"):
                    old.unlink()
            print("ALLOCATION CAPTURE", number, flush=True)
        dist.barrier()
    if rank == 0:
        write(
            work / "allocation_capture_complete.json",
            {
                "complete": True,
                "tokens": 65536,
                "training_sequence_ids": sequences.tolist(),
                "trajectory": "native released weights",
                "validation_or_audit_used": False,
            },
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
