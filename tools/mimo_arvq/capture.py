# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rolling, data-parallel MiMo captures for the unchanged expert-parallel PV."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent / "recipe"))
from arvq88.activation import activation_ste
from arvq88.inputs import write
from arvq88.pv import Projection
from propagation_math import expert_from_packed_input
from reference import Layer
from source import Source


def load_tokens(work, split):
    name = "train.npy" if split == "train" else f"{split}_eval.npy"
    return np.load(work / "corpus" / name, mmap_mode="r")


def batches(tokens, rank, world, train, size=None):
    # Fixed, disjoint validation/audit sets; training traverses the whole corpus.
    size = int(os.environ.get("MIMO_CAPTURE_BATCH", "4")) if size is None else size
    total = len(tokens) if train else min(16384, len(tokens))
    seqs = list(range((total + 1023) // 1024))[rank::world]
    for start in range(0, len(seqs), size):
        numbers = seqs[start : start + size]
        lengths = [min(1024, total - s * 1024) for s in numbers]
        ids = np.full((len(numbers), 1024), 151643, dtype=np.int64)
        for i, (s, length) in enumerate(zip(numbers, lengths)):
            ids[i, :length] = tokens[s * 1024 : s * 1024 + length]
        yield numbers, lengths, torch.from_numpy(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--mode", choices=["capture", "propagate"], required=True)
    args = parser.parse_args()
    work, number = args.work, args.layer
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    dev = f"cuda:{rank}"
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    source = Source(work / "source")
    layer = Layer(source, number, dev)
    cfg = json.loads((work / "config.json").read_text())
    train_root = work / f"training_capture{number}"
    eval_root = work / f"capture{number}"
    state_root = work / f"states{number}"
    out_root = work / f"states{number + 1}"
    for root in (train_root, eval_root, out_root):
        root.mkdir(exist_ok=True)
    if args.mode == "capture":
        layer.load_experts()
        if number == 1:
            dense = Layer(source, 0, dev)
            embedding = source.tensor("model.embed_tokens.weight", dev).to(
                torch.bfloat16
            )
        evaluations = []
        entries = []
        coverage = torch.zeros(384, dtype=torch.int64)
        energy = 0.0
        rows_total = 0
        for split, label in (("train", 0), ("validation", 1), ("audit", 2)):
            tokens = load_tokens(work, split)
            for chunk, (numbers, lengths, token_ids) in enumerate(
                batches(tokens, rank, world, split == "train")
            ):
                name = f"{split}_rank{rank}_{chunk:05d}.pt"
                ids_gpu = token_ids.to(dev)
                with torch.no_grad():
                    if number == 1:
                        state = dense.dense_forward(
                            torch.nn.functional.embedding(ids_gpu, embedding)
                        )
                    else:
                        cached = torch.load(
                            state_root / name, map_location=dev, weights_only=True
                        )
                        if cached["sequence_ids"] != numbers:
                            raise ValueError("Trajectory sequence identity mismatch")
                        state = cached["state"]
                    residual, x = layer.front(state)
                    flat = torch.cat([x[i, :n] for i, n in enumerate(lengths)])
                    ids, gates = layer.route(flat)
                    target = layer.native_moe(flat, ids, gates)
                    following = []
                    for sequence, n in zip(numbers, lengths):
                        start = sequence * 1024 + 1
                        next_ids = np.full(n, 151643, dtype=np.int64)
                        actual = tokens[start : min(start + n, len(tokens))]
                        next_ids[: len(actual)] = actual
                        following.append(torch.from_numpy(next_ids).to(dev))
                    following = torch.cat(following)
                    weights = torch.ones(len(flat), device=dev)
                    boundary = torch.isin(
                        following, torch.tensor(cfg["boundary_token_ids"], device=dev)
                    )
                    weights[boundary] = cfg["boundary_boost"]
                    data = {
                        "x": flat.cpu(),
                        "topk_ids": ids.cpu(),
                        "topk_weights": gates.cpu(),
                        "required": target.cpu(),
                        "row_weight": weights.cpu(),
                        "sequence_ids": numbers,
                        "sequence_lengths": lengths,
                        "residual": residual.cpu(),
                    }
                torch.save(data, train_root / name)
                if split == "train":
                    entries.append(
                        {
                            "file": name,
                            "sequence_ids": numbers,
                            "sequence_lengths": lengths,
                            "rows": len(flat),
                        }
                    )
                    coverage += torch.bincount(ids.cpu().flatten(), minlength=384)
                    energy += float(target.double().square().sum())
                    rows_total += len(flat)
                if split != "train" or chunk < 2:
                    evaluations.append(
                        {
                            "x": data["x"],
                            "topk_ids": data["topk_ids"],
                            "topk_weights": data["topk_weights"],
                            "pv_split": torch.full((len(flat),), label),
                            "frozen_output": torch.zeros_like(data["required"]),
                            "reference_output": data["required"],
                            "same_input_output": data["required"],
                            "row_weight": data["row_weight"],
                            "sequence_ids": torch.tensor(numbers).repeat_interleave(
                                torch.tensor(lengths)
                            ),
                        }
                    )
                if rank == 0 and chunk % 25 == 0:
                    print("CAPTURE", number, split, chunk, flush=True)
        torch.save(
            {
                key: torch.cat([part[key] for part in evaluations])
                for key in evaluations[0]
            },
            eval_root / f"rank{rank}.pt",
        )
        write(
            train_root / f"rank{rank}_complete.json",
            {
                "complete": True,
                "total_corpus_tokens": len(load_tokens(work, "train")),
                "rows": rows_total,
                "required_energy": energy,
                "coverage": coverage.tolist(),
                "entries": entries,
            },
        )
    else:
        # The reconstruction used here is the exact same export replay as PV.
        selected = work / f"layer{number}_same_input" / "merged"
        store = {}
        for key in ("w13", "w2"):
            store.update(
                torch.load(selected / f"{key}.pt", weights_only=True, mmap=True)
            )
        p13 = Projection(store, "", "w13", 384, dev)
        p2 = Projection(store, "", "w2", 384, dev)
        decoded = [(p13.weight(e).detach(), p2.weight(e).detach()) for e in range(384)]
        del p13, p2, store
        for split in ("train", "validation", "audit"):
            for path in sorted(train_root.glob(f"{split}_rank{rank}_*.pt")):
                data = torch.load(path, map_location=dev, weights_only=True)
                x, ids, gates = (
                    data["x"].float(),
                    data["topk_ids"],
                    data["topk_weights"],
                )
                y = torch.zeros_like(x)
                with torch.no_grad():
                    packed_input = activation_ste(x)
                    for e, (w13, w2) in enumerate(decoded):
                        rows, slots = torch.where(ids == e)
                        if len(rows):
                            output = expert_from_packed_input(
                                packed_input[rows], w13, w2
                            )
                            y.index_add_(0, rows, output * gates[rows, slots, None])
                    state = data["residual"]
                    offset = 0
                    for i, length in enumerate(data["sequence_lengths"]):
                        state[i, :length] += y[offset : offset + length].to(state.dtype)
                        offset += length
                torch.save(
                    {"state": state.cpu(), "sequence_ids": data["sequence_ids"]},
                    out_root / path.name,
                )
        write(
            out_root / f"rank{rank}_complete.json",
            {"complete": True, "parent_layer": number},
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
