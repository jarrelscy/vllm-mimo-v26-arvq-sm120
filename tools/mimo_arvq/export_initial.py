# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export all prepared hybrid initial fits without another training pass."""

import argparse
import json
import os
from pathlib import Path

import torch
from driver import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    rank, world = (
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )
    torch.set_num_threads(4)
    if not (work / "hybrid_prepared.json").exists():
        raise ValueError("Hybrid preparation must finish before export")
    for layer in range(1 + rank, 70, world):
        ready = work / "exports/initial" / f"layer{layer}/ready.json"
        if not ready.exists():
            export(work, layer, "initial")
        (work / f"export_initial_rank{rank}.json").write_text(
            json.dumps(
                {"last_exported_layer": layer, "phase": "initial", "training": False}
            )
        )
        print("INITIAL EXPORTED", layer, flush=True)


if __name__ == "__main__":
    main()
