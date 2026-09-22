# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fix a category-balanced held-out set before any layer is fitted."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.work / "corpus"
    if not (root / "complete.json").exists():
        raise ValueError("Corpus build is incomplete")
    entries = [json.loads(line) for line in (root / "documents_manifest.jsonl").open()]
    allocations = {
        "code": 3,
        "agentic": 3,
        "reasoning": 4,
        "instruction": 2,
        "medical": 2,
        "prose": 2,
    }
    selected = {}
    for split in ("validation", "audit"):
        raw = np.load(root / f"{split}.npy", mmap_mode="r")
        arrays = []
        selected[split] = []
        for category, count in allocations.items():
            candidates = sorted(
                (
                    e
                    for e in entries
                    if e["split"] == split and e["category"] == category
                ),
                key=lambda e: e["id"],
            )
            remaining = count * 1024
            for entry in candidates:
                length = min(entry["tokens"], remaining)
                arrays.append(raw[entry["offset"] : entry["offset"] + length].copy())
                selected[split].append(entry)
                remaining -= length
                if remaining == 0:
                    break
            if remaining:
                raise ValueError(f"Insufficient {split}/{category} tokens")
        array = np.concatenate(arrays)
        np.save(root / f"{split}_eval.npy", array)
        selected[f"{split}_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
    train_ids = {e["id"] for e in entries if e["split"] == "train"}
    for split in ("validation", "audit"):
        if train_ids & {e["id"] for e in selected[split]}:
            raise ValueError("Train/eval document leakage")
    (root / "evaluation_selection.json").write_text(json.dumps(selected, indent=2))
    print("Fixed 16K validation and 16K audit, each covering all six categories")


if __name__ == "__main__":
    main()
