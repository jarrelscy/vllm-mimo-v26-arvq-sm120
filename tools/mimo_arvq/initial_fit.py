# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-parallel Hessian initial fit using the pinned ARVQ encoder."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent / "recipe"))
from arvq88.encoder import fit_layer_projection, hessian_fc1, hessian_fc2
from arvq88.inputs import write
from source import Source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    dev = f"cuda:{rank}"
    work, layer = args.work, args.layer
    source = Source(work / "source")
    root = work / "baseline" / "initial" / f"layer_{layer:05d}"
    root.mkdir(parents=True, exist_ok=True)
    parts = []
    for owner in range(world):
        all_files = sorted(
            (work / f"training_capture{layer}").glob(f"train_rank{owner}_*.pt")
        )
        files = [
            all_files[i]
            for i in torch.linspace(0, len(all_files) - 1, 8).long().unique().tolist()
        ]
        parts.extend(torch.load(p, weights_only=True, mmap=True) for p in files)
    x = torch.cat([p["x"] for p in parts]).float().to(dev)
    ids = torch.cat([p["topk_ids"] for p in parts]).to(dev)
    del parts
    experts = list(range(rank, 384, world))
    weights = {
        e: tuple(
            source.expert(layer, e, p, dev).half()
            for p in ("gate_proj", "up_proj", "down_proj")
        )
        for e in experts
    }
    globals_ = {}
    for projection in ("w13", "w2"):
        samples = []
        for e in experts:
            wg, wu, wd = weights[e]
            w = torch.cat((wg, wu)) if projection == "w13" else wd
            rms = (
                w.float()
                .reshape(w.shape[0], -1, 128)
                .square()
                .mean(-1)
                .sqrt()
                .flatten()
            )
            generator = torch.Generator(device=dev).manual_seed(e)
            samples.append(
                rms[torch.randperm(len(rms), generator=generator, device=dev)[:4096]]
            )
        local = torch.cat(samples)
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        globals_[projection] = float(torch.cat(gathered).median().clamp_min(1e-8))
    started = time.time()
    for e in experts:
        destination = root / f"expert_{e:03d}.pt"
        if destination.exists():
            saved = torch.load(destination, weights_only=True, mmap=True)
            if saved["globals"] != globals_:
                raise ValueError("Initial-fit global scale identity changed")
            continue
        wg, wu, wd = weights[e]
        routed = x[(ids == e).any(-1)][:4096]
        if len(routed) < 32:
            routed = x[:4096]
        results = {}
        for projection in ("w13", "w2"):
            weight = torch.cat((wg, wu)) if projection == "w13" else wd
            hessian = (
                hessian_fc1(routed)
                if projection == "w13"
                else hessian_fc2(routed, wg, wu)
            )
            encoded = fit_layer_projection(
                [weight],
                [hessian],
                device=dev,
                global_scale=globals_[projection],
                codebook_scope="layer",
                seed=e,
                cb_iters=8,
                sweep_passes=2,
                col_block=128,
                refine=1,
                subsample_per_expert=20000,
            )
            results[projection] = vars(encoded)
            del hessian, encoded
        temporary = destination.with_suffix(".tmp")
        torch.save({"expert": e, "globals": globals_, "encoded": results}, temporary)
        temporary.replace(destination)
        print("INITIAL", layer, rank, e, round(time.time() - started, 1), flush=True)
    dist.barrier()
    if rank == 0:
        for projection in ("w13", "w2"):
            values = [
                torch.load(root / f"expert_{e:03d}.pt", weights_only=True, mmap=True)[
                    "encoded"
                ][projection]
                for e in range(384)
            ]
            merged = dict(values[0])
            for key in ("c0", "c1"):
                merged[key] = torch.stack([v[key] for v in values])
            for key in ("a", "b", "s"):
                merged[key] = torch.cat([v[key] for v in values])
            merged["scale_dtype"] = "fp16"
            merged["s"] = merged["s"].half().float()
            torch.save({projection: merged}, root / f"{projection}.pt")
        write(
            root / "arvq-manifest.json",
            {
                "layer": layer,
                "cold_expert_ids": list(range(384)),
                "fit": "per-expert Hessian initial fit; FP16 block-scale container",
                "codebook_scope": "expert",
                "no_rotation": True,
                "source_revision": "73875d00b30a89ef8cc353a0b60b0e9f9561952d",
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
