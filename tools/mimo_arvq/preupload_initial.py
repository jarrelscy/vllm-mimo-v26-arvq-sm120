# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preupload immutable weights concurrently; publisher owns all repo commits."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from publish import REPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    work = args.work
    state = json.loads((work / "upload_state.json").read_text())
    root = work / "preupload"
    root.mkdir(exist_ok=True)

    def upload(layer):
        if str(layer) in state["layers"] or (root / f"{layer}.json").exists():
            return
        export = work / "exports/initial" / f"layer{layer}"
        if not (export / "ready.json").exists():
            raise ValueError(f"Layer {layer} is not ready")
        manifest = json.loads((export / f"layer-{layer:03d}-manifest.json").read_text())
        paths = [(v["file"], export / v["file"]) for v in manifest["files"].values()]
        paths.append(
            (
                f"roster-layer-{layer:03d}.safetensors",
                work / "hot" / f"layer{layer}.safetensors",
            )
        )
        for attempt in range(3):
            try:
                additions = [
                    CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(p))
                    for name, p in paths
                ]
                HfApi().preupload_lfs_files(REPO, additions, num_threads=3)
                (root / f"{layer}.json").write_text(
                    json.dumps(
                        {"preuploaded": True, "committed": False, "time": time.time()}
                    )
                )
                print("PREUPLOADED", layer, flush=True)
                return
            except Exception as error:
                print("PREUPLOAD RETRY", layer, attempt, repr(error), flush=True)
                time.sleep(10)
        raise RuntimeError(f"Preupload failed for {layer}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, layer) for layer in range(1, 70)]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
