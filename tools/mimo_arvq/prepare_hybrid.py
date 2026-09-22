# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score all fitted candidates on training probes, allocate, seed hybrid PV."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "recipe"))
from arvq88.activation import activation_ste
from arvq88.inputs import write
from arvq88.pv import Projection
from hybrid import export_hot, quantized_expert
from propagation_math import expert_from_packed_input
from source import Source


def candidate_source(parent, work, layer):
    for path in (
        parent / f"layer{layer}_same_input/merged",
        parent / "baseline/initial" / f"layer_{layer:05d}",
        work / "allocation_calibration/baseline/initial" / f"layer_{layer:05d}",
    ):
        if (path / "arvq-manifest.json").exists():
            return path
    raise ValueError(f"Missing candidate fit for layer {layer}")


def select_hot(scores, count, cap=192):
    """Global benefit ranking with deterministic ties and a per-layer cap."""
    selected = {str(layer): [] for layer in range(1, 70)}
    ordered = sorted(
        (
            (-float(score), layer, e)
            for layer, values in scores.items()
            for e, score in enumerate(values)
        )
    )
    for _, layer, expert in ordered:
        if len(selected[str(layer)]) < cap:
            selected[str(layer)].append(expert)
            count -= 1
            if count == 0:
                return {k: sorted(v) for k, v in selected.items()}
    raise ValueError("Insufficient eligible hot slots")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--score-layers", type=int, nargs="+")
    args = p.parse_args()
    parent, work = args.parent, args.work
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    dev = f"cuda:{rank}"
    source = Source(parent / "source")
    score_root = work / "allocation_scores"
    score_root.mkdir(parents=True, exist_ok=True)
    calibration = work / "allocation_calibration"
    if not (calibration / "allocation_capture_complete.json").exists():
        calibration = parent
    (work / "hot").mkdir(exist_ok=True)
    for layer in range(1 + rank, 70, world):
        if args.score_layers and layer not in args.score_layers:
            continue
        path = score_root / f"layer{layer}.json"
        if path.exists():
            continue
        selected = candidate_source(parent, work, layer)
        store = {}
        for key in ("w13", "w2"):
            store.update(
                torch.load(selected / f"{key}.pt", weights_only=True, mmap=True)
            )
        p13, p2 = [Projection(store, "", key, 384, dev) for key in ("w13", "w2")]
        parts = [
            torch.load(
                calibration / f"capture{layer}/rank{r}.pt", weights_only=True, mmap=True
            )
            for r in range(world)
        ]
        data = {
            k: torch.cat([v[k][v["pv_split"] == 0] for v in parts]).to(dev)
            for k in ("x", "topk_ids", "topk_weights")
        }
        x = data["x"].float()
        ids, gates = data["topk_ids"], data["topk_weights"]
        packed = activation_ste(x)
        scores, counts = [], []
        for expert in range(384):
            rows, slots = torch.where(ids == expert)
            counts.append(len(rows))
            if not len(rows):
                scores.append(0.0)
                continue
            _, (h13, h2) = quantized_expert(source, layer, expert, dev)
            wg, wu, wd = [
                source.expert(layer, expert, key, dev)
                for key in ("gate_proj", "up_proj", "down_proj")
            ]
            cold = expert_from_packed_input(
                packed[rows], p13.weight(expert), p2.weight(expert)
            )
            hot = expert_from_packed_input(packed[rows], h13, h2)
            # Match the source reference's BF16 expert linear boundaries.
            z = x[rows].bfloat16()
            g, u = F.linear(z, wg.bfloat16()), F.linear(z, wu.bfloat16())
            ref = F.linear(F.silu(g) * u, wd.bfloat16()).float()
            benefit = (cold.double() - ref).square().sum(-1) - (
                hot.double() - ref
            ).square().sum(-1)
            score = (benefit * gates[rows, slots].double().square()).sum()
            if not torch.isfinite(score):
                raise ValueError("Nonfinite allocation score")
            scores.append(float(score))
        write(
            path,
            {
                "layer": layer,
                "scores": scores,
                "routed_rows": counts,
                "training_rows": len(x),
                "source": str(selected),
                "metric": "routing-weighted cold SSE minus NVFP4 SSE",
                "scope": "training probes only; emulated serving arithmetic",
            },
        )
        del p13, p2, store, parts, data, x, ids, gates, packed
        print("SCORED", layer, flush=True)
    dist.barrier()
    if args.score_layers:
        dist.destroy_process_group()
        return
    if rank == 0:
        scores = {
            layer: json.loads((score_root / f"layer{layer}.json").read_text())["scores"]
            for layer in range(1, 70)
        }
        hot = select_hot(scores, 1325)
        write(
            work / "allocation.json",
            {
                "hot_count": 1325,
                "total_experts": 26496,
                "fraction": 1325 / 26496,
                "per_layer_cap": 192,
                "method": "global ARVQ-specific routing-weighted output-error benefit",
                "limitations": (
                    "text training probes; additive expert proxy; "
                    "no cross-expert covariance"
                ),
                "layers": {k: {"hot": v} for k, v in hot.items()},
            },
        )
    dist.barrier()
    plan = json.loads((work / "allocation.json").read_text())
    for layer in range(1 + rank, 70, world):
        hot = plan["layers"][str(layer)]["hot"]
        cold = [e for e in range(384) if e not in hot]
        root = work / "baseline/initial" / f"layer_{layer:05d}"
        root.mkdir(parents=True, exist_ok=True)
        selected = candidate_source(parent, work, layer)
        for key in ("w13", "w2"):
            data = torch.load(selected / f"{key}.pt", weights_only=True, mmap=True)[key]
            for name in ("c0", "c1", "a", "b", "s"):
                data[name] = data[name][cold].clone()
            torch.save({key: data}, root / f"{key}.pt")
        manifest = json.loads((selected / "arvq-manifest.json").read_text())
        manifest.update(
            cold_expert_ids=cold,
            hot_expert_ids=hot,
            fit="warm start from accepted all-cold PV, frozen NVFP4 hot experts",
        )
        write(root / "arvq-manifest.json", manifest)
        export_hot(source, layer, hot, work / "hot" / f"layer{layer}.safetensors", dev)
        write(
            work / "receipts" / f"initial{layer}.json",
            {"complete": True, "warm_start": str(selected)},
        )
    dist.barrier()
    if rank == 0:
        cfg = json.loads((parent / "config.json").read_text())
        cfg.update(
            baseline=str(work / "baseline"),
            corpus=str(work / "corpus"),
            hot_fraction=1325 / 26496,
            allocation_status="global output-benefit allocation; frozen during PV",
        )
        write(work / "config.json", cfg)
        for name in ("source", "corpus"):
            path = work / name
            if not path.exists():
                path.symlink_to(parent / name, target_is_directory=True)
        for name in ("source_verified.json", "capture_attention_qualified.json"):
            shutil.copy2(parent / name, work / name)
        from publish import seed

        root = seed(work)
        shutil.copy2(
            parent / "seed/model.safetensors.index.json",
            root / "model.safetensors.index.json",
        )
        (root / "README.md").write_text("""---
license: mit
base_model: XiaomiMiMo/MiMo-V2.6-Pro-RL
library_name: vllm
---
# MiMo-V2.6-Pro-RL ARVQ / NVFP4 hybrid

Work in progress: replacing the all-ARVQ candidate checkpoint with a 5% hot
hybrid (1325 of 26496 routed experts). Allocation ranks measured training-only
output-error benefit of NVFP4 over the accepted per-expert ARVQ candidates.
Each layer receives another sequential PV pass with its NVFP4 hot experts
frozen, followed by held-out validation, audit and serialization checks.
Weights, roster and config change atomically per layer; config counts describe
the currently uploaded mixture. See allocation.json and pv_progress.json.

All backbone, MTP, vision, audio, audio-tokenizer and DFlash weights are retained.
Text-only calibration; emulated arithmetic. Actual SM120 serving, multimodal
quality and the 1M-context memory budget remain unqualified.
""")
        # Keep the HF backbone; replace each layer's roster/config atomically.
        write(work / "upload_state.json", {"seed": True, "layers": {}})
        write(work / "hybrid_prepared.json", {"complete": True, "hot_count": 1325})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
