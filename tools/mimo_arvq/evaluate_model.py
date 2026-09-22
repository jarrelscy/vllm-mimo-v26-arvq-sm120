# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent held-out, layer-streamed end-to-end reconstruction evaluation.

This evaluates the calibration arithmetic, not the SM120 serving executable.
It does not select checkpoints or feed results back into PV.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "recipe"))
from arvq88.activation import ARITHMETIC
from arvq88.gradient_indices import expert_output
from arvq88.inputs import write
from arvq88.pv import Projection
from hybrid import allocation, load_hot
from reference import Layer, rms
from source import Source


def prepare_test(work):
    root = work / "corpus"
    destination = root / "final_test_selection.json"
    if destination.exists():
        return
    selected = json.loads((root / "evaluation_selection.json").read_text())
    used = {e["id"] for split in ("validation", "audit") for e in selected[split]}
    entries = [json.loads(line) for line in (root / "documents_manifest.jsonl").open()]
    raw = np.load(root / "audit.npy", mmap_mode="r")
    arrays, chosen = [], []
    for category, count in dict(
        code=3, agentic=3, reasoning=4, instruction=2, medical=2, prose=2
    ).items():
        remaining = count * 1024
        candidates = sorted(
            (
                e
                for e in entries
                if e["split"] == "audit"
                and e["category"] == category
                and e["id"] not in used
            ),
            key=lambda e: e["id"],
        )
        for entry in candidates:
            take = min(remaining, entry["tokens"])
            arrays.append(raw[entry["offset"] : entry["offset"] + take].copy())
            chosen.append(entry)
            remaining -= take
            if not remaining:
                break
        if remaining:
            raise ValueError(f"Insufficient independent test data: {category}")
    array = np.concatenate(arrays)
    assert not {e["id"] for e in chosen} & used
    np.save(root / "final_test.npy", array)
    write(
        destination,
        {
            "tokens": len(array),
            "documents": chosen,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "selection": "unused audit documents, no checkpoint selection",
        },
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    work = args.work
    if args.prepare_only:
        prepare_test(work)
        return
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    dev = f"cuda:{rank}"
    root = work / "full_model_evaluation"
    root.mkdir(exist_ok=True)
    if rank == 0:
        prepare_test(work)
    dist.barrier()
    raw = np.load(work / "corpus/final_test.npy").reshape(-1, 1024)
    tokens = torch.tensor(raw[rank::world].astype(np.int64), device=dev)
    source = Source(work / "source")
    completed = sorted(root.glob("layer_*_complete.json"))
    start = 0
    if completed:
        start = json.loads(completed[-1].read_text())["layer"] + 1
        saved = torch.load(
            root / f"layer_{start - 1:03d}_rank{rank}.pt",
            weights_only=True,
            map_location=dev,
        )
        teacher, student = saved["teacher"], saved["student"]
    else:
        embedding = source.tensor("model.embed_tokens.weight", dev).bfloat16()
        teacher = F.embedding(tokens, embedding)
        student = teacher.clone()
        del embedding
    for number in range(start, 70):
        if number and not (work / "exports/pv" / f"layer{number}/ready.json").exists():
            raise ValueError(f"Missing accepted export for layer {number}")
        layer = Layer(source, number, dev)
        if number == 0:
            teacher = layer.dense_forward(teacher)
            student = teacher.clone()
        else:
            residual, x = layer.front(teacher)
            flat = x.flatten(0, 1)
            ids, gates = layer.route(flat)
            output = layer.native_moe(flat, ids, gates)
            teacher = residual + output.reshape_as(residual).to(residual.dtype)
            layer.experts = None
            selected = work / f"layer{number}_same_input/merged"
            store = {}
            for key in ("w13", "w2"):
                store.update(
                    torch.load(selected / f"{key}.pt", weights_only=True, mmap=True)
                )
            _, cold = allocation(work, number)
            hot_weights = load_hot(work, number, dev)
            p13, p2 = [
                Projection(store, "", key, len(cold), dev) for key in ("w13", "w2")
            ]
            cold_slots = {expert: slot for slot, expert in enumerate(cold)}
            residual, x = layer.front(student)
            flat = x.flatten(0, 1).float()
            ids, gates = layer.route(flat)
            output = torch.zeros_like(flat)
            for e in range(384):
                rows, slots = torch.where(ids == e)
                if len(rows):
                    if e in hot_weights:
                        w13, w2 = hot_weights[e]
                    else:
                        slot = cold_slots[e]
                        w13, w2 = p13.weight(slot), p2.weight(slot)
                    y = expert_output(flat[rows], w13, w2, ARITHMETIC)
                    output.index_add_(0, rows, y * gates[rows, slots, None])
            student = residual + output.reshape_as(residual).to(residual.dtype)
            del p13, p2, store, residual, x, flat, output, ids, gates
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise ValueError(f"Nonfinite full-model hidden states at layer {number}")
        del layer
        torch.save(
            {"teacher": teacher.cpu(), "student": student.cpu()},
            root / f"layer_{number:03d}_rank{rank}.pt",
        )
        dist.barrier()
        if rank == 0:
            write(root / f"layer_{number:03d}_complete.json", {"layer": number})
            print("FULL MODEL", number, flush=True)
    norm = source.tensor("model.norm.weight", dev).bfloat16()
    head = source.tensor("lm_head.weight", dev).bfloat16()
    teacher = rms(teacher, norm)[:, :-1].flatten(0, 1)
    student = rms(student, norm)[:, :-1].flatten(0, 1)
    labels = tokens[:, 1:].flatten()
    sums = torch.zeros(4, dtype=torch.float64, device=dev)
    for offset in range(0, len(labels), 128):
        sl = slice(offset, offset + 128)
        t = F.log_softmax(F.linear(teacher[sl], head).float(), -1)
        s = F.log_softmax(F.linear(student[sl], head).float(), -1)
        sums[0] += F.nll_loss(t, labels[sl], reduction="sum").double()
        sums[1] += F.nll_loss(s, labels[sl], reduction="sum").double()
        sums[2] += (t.exp() * (t - s)).sum().double()
        sums[3] += len(labels[sl])
    dist.all_reduce(sums)
    if not torch.isfinite(sums).all():
        raise ValueError("Nonfinite full-model likelihood")
    if rank == 0:
        nll_t, nll_s, kl = (sums[:3] / sums[3]).tolist()
        write(
            root / "report.json",
            {
                "complete": True,
                "tokens_scored": int(sums[3]),
                "teacher_nll": nll_t,
                "student_nll": nll_s,
                "teacher_perplexity": math.exp(nll_t),
                "student_perplexity": math.exp(nll_s),
                "excess_nll": nll_s - nll_t,
                "teacher_to_student_kl": kl,
                "scope": "Full-model emulation; serving parity unqualified",
                "test_selection": json.loads(
                    (work / "corpus/final_test_selection.json").read_text()
                ),
                "production_ready": False,
            },
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
