# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify complete remote weight hashes, index and allocation after upload."""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from publish import REPO, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--require-pv-layers", type=int, nargs="*", default=[])
    args = parser.parse_args()
    work = args.work
    while True:
        state = json.loads((work / "upload_state.json").read_text())
        if state.get("initial_upload_complete") and all(
            state["layers"].get(str(layer), {}).get("phase") == "pv"
            for layer in args.require_pv_layers
        ):
            break
        if not args.wait:
            raise ValueError("Upload is not complete")
        time.sleep(30)
    api = HfApi()
    info = api.model_info(REPO, files_metadata=True)
    files = {f.rfilename: f for f in info.siblings}
    config = json.loads(
        Path(hf_hub_download(REPO, "config.json", revision=info.sha)).read_text()
    )
    index = json.loads(
        Path(
            hf_hub_download(REPO, "model.safetensors.index.json", revision=info.sha)
        ).read_text()
    )["weight_map"]
    if set(state["layers"]) != {str(i) for i in range(1, 70)}:
        raise ValueError("Layer coverage is incomplete")
    checked = 0
    for layer in state["layers"].values():
        for name, digest in layer["sha256"].items():
            if (
                name not in files
                or files[name].lfs is None
                or files[name].lfs.sha256 != digest
            ):
                raise ValueError(f"Remote weight hash mismatch: {name}")
            checked += 1
    for path in (work / "seed").rglob("*.safetensors"):
        name = str(path.relative_to(work / "seed"))
        if (
            name not in files
            or files[name].lfs is None
            or files[name].lfs.sha256 != sha(path)
        ):
            raise ValueError(f"Preserved backbone/auxiliary mismatch: {name}")
        checked += 1
    if any(name not in files for name in index.values()):
        raise ValueError("Index refers to missing files")
    books = config["quantization_config"]["aqlm_layer_books"]
    if sum(v["n_nvfp4"] for v in books.values()) != 1325:
        raise ValueError("Hot budget mismatch")
    for layer in range(1, 70):
        item = books[str(layer)]
        if item["n_cold"] + item["n_nvfp4"] != 384:
            raise ValueError(f"Expert count mismatch at layer {layer}")
    report = {
        "complete": True,
        "revision_verified": info.sha,
        "weight_files_hash_verified": checked,
        "moe_layers": 69,
        "hot_experts": 1325,
        "current_files_gib": sum(f.size or 0 for f in files.values()) / 2**30,
        "serving_and_quality_qualified": False,
    }
    path = work / "upload_verification.json"
    path.write_text(json.dumps(report, indent=2))
    api.upload_file(
        repo_id=REPO,
        path_in_repo="upload_verification.json",
        path_or_fileobj=str(path),
        commit_message="Verify complete hybrid weight upload and preserved auxiliaries",
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
