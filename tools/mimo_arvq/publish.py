# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish immutable per-layer initial/PV exports with an atomic index update."""

import argparse
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            result.update(chunk)
    return result.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def seed(work):
    source, seed = work / "source", work / "seed"
    seed.mkdir(exist_ok=True)
    if (seed / "seed_ready.json").exists():
        return seed
    original = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    by_file = {}
    for key, filename in original.items():
        if ".mlp.experts." not in key:
            by_file.setdefault(filename, []).append(key)
    index = {}
    for i, (filename, keys) in enumerate(by_file.items()):
        name = f"backbone-{i:03d}.safetensors"
        if not (seed / name).exists():
            with safe_open(source / filename, framework="pt") as handle:
                tensors = {key: handle.get_tensor(key) for key in keys}
            save_file(tensors, seed / name)
        index.update({key: name for key in keys})
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if (
            not path.is_file()
            or ".cache" in relative.parts
            or relative.parts[0].startswith(".")
        ):
            continue
        if len(relative.parts) == 1 and (
            path.suffix == ".safetensors"
            or path.name in ("README.md", "config.json", "model.safetensors.index.json")
        ):
            continue
        destination = seed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
    config = json.loads((source / "config.json").read_text())
    original_quant = config["quantization_config"]
    config["quantization_config"] = {
        "quant_method": "nvfp4_arvq_hybrid",
        "source_fp8": original_quant,
        "nvfp4": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
        },
        "arvq": {
            "format": "rvq256_256x8_expert_fp16block",
            "version": 4,
            "activation_planes": 4,
            "weight_scale_group": 128,
            "codebook_scope": "expert",
            "codebook_sizes": [256, 256],
        },
        "aqlm_layer_books": {
            str(layer): {
                "n_nvfp4": 0,
                "n_base": 0,
                "n_cold": 384,
                "cold_expert_ids": list(range(384)),
            }
            for layer in range(1, 70)
        },
    }
    config["arvq_campaign"] = {
        "production_ready": False,
        "allocation": "all-cold memory baseline; no expert pruning",
        "source_revision": "73875d00b30a89ef8cc353a0b60b0e9f9561952d",
    }
    atomic(seed / "config.json", config)
    atomic(seed / "model.safetensors.index.json", {"metadata": {}, "weight_map": index})
    (seed / "README.md").write_text("""---
license: mit
base_model: XiaomiMiMo/MiMo-V2.6-Pro-RL
tags: [arvq, multimodal, work-in-progress]
---
# MiMo-V2.6-Pro-RL ARVQ

**Incremental initial fitting and PV are in progress. This is not a complete,
validated serving checkpoint until every layer is present and all gates pass.**

All vision/audio, audio tokenizer, embedded MTP and separate DFlash weights
are retained. Routed experts use independent FP4 codebooks and FP16 block
scales (2.125 bpw including scales). No expert pruning. This initial memory
baseline assigns all routed experts to ARVQ; non-expert FP8/BF16 weights are
preserved from the release. It does not claim a measured hot allocation.

PV uses same-input MoE output loss, Adam, activation-weighted discrete index
proposals and held-out checkpoint selection. Training uses native MiMo text
tokenization of the prior calibration-source mixture. Multimodal weights are
preserved but multimodal quality and SM120 1M-context serving are unverified.

See pv_progress.json for initial/PV coverage. Missing layers are not usable.
Serving fork: https://github.com/jarrelscy/vllm-mimo-v26-arvq-sm120
""")
    atomic(
        seed / "seed_ready.json",
        {"complete": True, "preserved_auxiliary_weights": True},
    )
    return seed


def roster(layer, root, work=None):
    if work is not None and (work / "allocation.json").exists():
        path = root / f"roster-layer-{layer:03d}.safetensors"
        shutil.copy2(work / "hot" / f"layer{layer}.safetensors", path)
        return path
    prefix = f"model.layers.{layer}.mlp.experts."
    tensors = {prefix + "hyb_kind": torch.full((384,), 2, dtype=torch.int8)}
    shapes = {
        "nvfp4_w13_packed": (0, 4096, 3072),
        "nvfp4_w13_bscale": (0, 4096, 384),
        "nvfp4_w13_scale2": (0, 2),
        "nvfp4_w2_packed": (0, 6144, 1024),
        "nvfp4_w2_bscale": (0, 6144, 128),
        "nvfp4_w2_scale2": (0, 1),
    }
    for name, shape in shapes.items():
        tensors[prefix + name] = torch.empty(
            shape, dtype=torch.float32 if name.endswith("scale2") else torch.uint8
        )
    path = root / f"roster-layer-{layer:03d}.safetensors"
    save_file(tensors, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    lock = (work / "publisher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    api = HfApi()
    root = seed(work)
    state_path = work / "upload_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"seed": False, "layers": {}}
    )
    if not state["seed"]:
        api.upload_folder(
            repo_id=REPO,
            folder_path=root,
            commit_message="Preserve MiMo backbone and all auxiliary modules",
        )
        state["seed"] = True
        atomic(state_path, state)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    while not (work / "STOP_PUBLISH").exists():
        try:
            for layer in range(1, 70):
                for phase in ("initial", "pv"):
                    old = state["layers"].get(str(layer), {})
                    previous = old.get("phase")
                    if previous == "pv" and (
                        phase != "pv" or old.get("reports_uploaded", False)
                    ):
                        continue
                    if previous == phase and phase != "pv":
                        continue
                    export = work / "exports" / phase / f"layer{layer}"
                    ready = export / "ready.json"
                    if not ready.exists():
                        continue
                    manifest = json.loads(
                        (export / f"layer-{layer:03d}-manifest.json").read_text()
                    )
                    paths = [
                        export / item["file"] for item in manifest["files"].values()
                    ]
                    paths.append(roster(layer, export, work))
                    expected = {}
                    for path in paths:
                        expected[path.name] = sha(path)
                        with safe_open(path, framework="pt") as handle:
                            tensor_names = handle.keys()
                            index["weight_map"].update(
                                {key: path.name for key in tensor_names}
                            )
                    layer_state = {
                        "phase": phase,
                        "sha256": expected,
                        "reports_uploaded": phase == "pv",
                    }
                    if phase == "pv":
                        report = json.loads((export / "pv_report.json").read_text())
                        selection = json.loads(
                            (
                                work / f"layer{layer}_same_input" / "selection.json"
                            ).read_text()
                        )
                        retained_initial = selection["selection"] == "initial"
                        layer_state.update(
                            {
                                "selection": selection["selection"],
                                "best_step": selection["best_step"],
                                "initial_validation": report["initial_validation"][
                                    "target_rel"
                                ],
                                "selected_validation": report[
                                    "initial_validation"
                                    if retained_initial
                                    else "final_validation"
                                ]["target_rel"],
                                "selected_audit": report[
                                    "initial_development_audit"
                                    if retained_initial
                                    else "development_audit"
                                ]["target_rel"],
                            }
                        )
                    new_layers = {
                        **state["layers"],
                        str(layer): layer_state,
                    }
                    progress = {
                        "layers": new_layers,
                        "total_moe_layers": 69,
                        "initial_layers": len(new_layers),
                        "pv_layers": sum(
                            v["phase"] == "pv" for v in new_layers.values()
                        ),
                        "production_ready": False,
                    }
                    operations = [
                        CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p))
                        for p in paths
                    ]
                    operations += [
                        CommitOperationAdd(
                            path_in_repo=f"cold_manifests/layer-{layer:03d}-manifest.json",
                            path_or_fileobj=str(
                                export / f"layer-{layer:03d}-manifest.json"
                            ),
                        ),
                        CommitOperationAdd(
                            path_in_repo="model.safetensors.index.json",
                            path_or_fileobj=json.dumps(index).encode(),
                        ),
                        CommitOperationAdd(
                            path_in_repo="pv_progress.json",
                            path_or_fileobj=json.dumps(progress, indent=2).encode(),
                        ),
                    ]
                    config = None
                    if (work / "allocation.json").exists():
                        config = json.loads((root / "config.json").read_text())
                        cold = manifest["cold_expert_ids"]
                        hot = manifest["hot_expert_ids"]
                        config["quantization_config"]["aqlm_layer_books"][
                            str(layer)
                        ] = {
                            "n_nvfp4": len(hot),
                            "n_base": 0,
                            "n_cold": len(cold),
                            "cold_expert_ids": cold,
                        }
                        config["arvq_campaign"]["allocation"] = (
                            "transitioning to 1325 NVFP4 hot experts; "
                            "per-layer counts authoritative"
                        )
                        operations.extend(
                            [
                                CommitOperationAdd(
                                    path_in_repo="config.json",
                                    path_or_fileobj=json.dumps(config).encode(),
                                ),
                                CommitOperationAdd(
                                    path_in_repo="allocation.json",
                                    path_or_fileobj=str(work / "allocation.json"),
                                ),
                                CommitOperationAdd(
                                    path_in_repo="README.md",
                                    path_or_fileobj=str(root / "README.md"),
                                ),
                            ]
                        )
                    if phase == "pv":
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"pv_reports/layer-{layer:03d}.json",
                                path_or_fileobj=str(export / "pv_report.json"),
                            )
                        )
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"pv_reports/layer-{layer:03d}-selection.json",
                                path_or_fileobj=str(
                                    work / f"layer{layer}_same_input" / "selection.json"
                                ),
                            )
                        )
                    api.create_commit(
                        repo_id=REPO,
                        operations=operations,
                        commit_message=f"Layer {layer}/69: {phase} ARVQ",
                    )
                    remote = {
                        f.rfilename: f
                        for f in api.model_info(REPO, files_metadata=True).siblings
                    }
                    for name, digest in expected.items():
                        item = remote[name]
                        if item.lfs is not None and item.lfs.sha256 != digest:
                            raise ValueError(f"Remote hash mismatch: {name}")
                    state["layers"] = new_layers
                    if config is not None:
                        atomic(root / "config.json", config)
                    atomic(root / "model.safetensors.index.json", index)
                    atomic(state_path, state)
                    print("UPLOADED", layer, phase, flush=True)
            if len(state["layers"]) == 69 and all(
                v["phase"] == "pv" for v in state["layers"].values()
            ):
                report = work / "full_model_evaluation/report.json"
                if report.exists():
                    operations = [
                        CommitOperationAdd(
                            path_in_repo="evaluation/full_model.json",
                            path_or_fileobj=str(report),
                        )
                    ]
                    for layer in range(1, 70):
                        for name in ("report", "selection"):
                            operations.append(
                                CommitOperationAdd(
                                    path_in_repo=f"pv_reports/layer-{layer:03d}-{name}.json",
                                    path_or_fileobj=str(
                                        work
                                        / f"layer{layer}_same_input"
                                        / f"{name}.json"
                                    ),
                                )
                            )
                    api.create_commit(
                        repo_id=REPO,
                        operations=operations,
                        commit_message="Publish final fitting quality report",
                    )
                    state["evaluation_uploaded"] = True
                    atomic(state_path, state)
                    return
        except Exception as error:
            atomic(
                work / "upload_error.json", {"error": str(error), "time": time.time()}
            )
            print("UPLOAD RETRY", repr(error), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
