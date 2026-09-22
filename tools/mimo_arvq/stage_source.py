# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage the complete, pinned MiMo release without loading model code."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SOURCE = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
REVISION = "73875d00b30a89ef8cc353a0b60b0e9f9561952d"


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    destination = args.work / "source"
    status = args.work / "source_status.json"
    receipt = args.work / "source_verified.json"
    manifest = args.work / "source_manifest.json"
    api = HfApi()
    started = time.time()
    try:
        info = api.model_info(SOURCE, revision=REVISION, files_metadata=True)
        files = []
        for item in info.siblings:
            lfs = item.lfs
            files.append(
                {
                    "path": item.rfilename,
                    "size": item.size,
                    "sha256": lfs.sha256 if lfs else None,
                }
            )
        write_json(manifest, {"repo": SOURCE, "revision": REVISION, "files": files})
        # Stage metadata first so tokenizer and architecture work can proceed.
        snapshot_download(
            SOURCE,
            revision=REVISION,
            local_dir=destination,
            ignore_patterns=["*.safetensors", "*.pt", "*.pdf"],
        )
        write_json(
            status,
            {
                "stage": "downloading",
                "started": started,
                "revision": REVISION,
                "expected_bytes": sum(f["size"] for f in files),
            },
        )
        snapshot_download(
            SOURCE, revision=REVISION, local_dir=destination, max_workers=8
        )
        verified = []
        for item in files:
            write_json(
                status,
                {
                    "stage": "verifying",
                    "started": started,
                    "file": item["path"],
                    "verified_files": len(verified),
                },
            )
            path = destination / item["path"]
            if path.stat().st_size != item["size"]:
                raise ValueError(f"Size mismatch: {item['path']}")
            if item["sha256"]:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while chunk := handle.read(16 * 1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"SHA256 mismatch: {item['path']}")
            verified.append(item["path"])
        result = {
            "stage": "complete",
            "repo": SOURCE,
            "revision": REVISION,
            "started": started,
            "completed": time.time(),
            "verified_files": len(verified),
            "preserved": [
                "vision",
                "audio",
                "audio_tokenizer",
                "embedded_mtp",
                "dflash",
            ],
        }
        write_json(receipt, result)
        write_json(status, result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(
            status,
            {
                "stage": "failed",
                "started": started,
                "error": str(error),
                "time": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
