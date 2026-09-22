# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild the previous raw-source mixture using MiMo's native template."""

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


class NativeTemplate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.last_record = None

    def apply_chat_template(self, messages, **kwargs):
        messages = copy.deepcopy(messages)
        for message in messages:
            content = message.get("content", "")
            if (
                message["role"] == "assistant"
                and isinstance(content, str)
                and "<think>" in content
                and "</think>" in content
            ):
                opening, tail = content.split("<think>", 1)
                reasoning, answer = tail.split("</think>", 1)
                message["reasoning_content"] = reasoning
                message["content"] = opening + answer
        kwargs.update(tokenize=False, add_generation_prompt=False)
        self.last_record = {"messages": messages}
        if "tools" in kwargs:
            self.last_record["tools"] = kwargs["tools"]
        return self.tokenizer.apply_chat_template(messages, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--legacy-tools", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=18_000_000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy_tools))
    import build_calib_v31 as old
    import build_calib_v32 as rebuilt
    import collect_expert_stats_v2 as generators
    import reasoning_slice

    root = args.work / "corpus"
    root.mkdir(exist_ok=True)
    if (root / "complete.json").exists():
        return
    tok = AutoTokenizer.from_pretrained(args.work / "source", trust_remote_code=False)
    adapter = NativeTemplate(tok)
    rng = random.Random(42)
    artifacts = generators.gather_artifacts(rng)
    mixes = {
        "code": (0.22, old.gen_code_hf()),
        "agentic": (0.15, old.gen_agentic(rng, artifacts, adapter)),
        "reasoning": (
            0.30,
            reasoning_slice.gen_reasoning(
                rng, adapter, clean_end=True, max_reason=3500, max_answer=1200
            ),
        ),
        "instruction": (0.13, old.gen_instruction(adapter)),
        "medical": (0.10, rebuilt.gen_medical(rng, adapter)),
        "prose": (0.10, rebuilt.gen_prose(adapter)),
    }
    counts = Counter()
    seen = set()
    handles = {
        s: (root / f"{s}.uint32").open("wb") for s in ("train", "validation", "audit")
    }
    raw = (root / "documents.jsonl").open("w")
    manifest = (root / "documents_manifest.jsonl").open("w")
    try:
        for category, (fraction, generator) in mixes.items():
            budget = int(args.tokens * fraction)
            for text in generator:
                if not text:
                    continue
                text = text[:200_000]
                identity = hashlib.sha256(text.encode()).hexdigest()
                if identity in seen:
                    continue
                seen.add(identity)
                bucket = int(identity[:8], 16) % 1000
                split = (
                    "validation" if bucket < 25 else "audit" if bucket < 50 else "train"
                )
                ids = tok.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                # Plain-text documents use the native document terminator.
                if ids[-1] not in (151643, 151645):
                    ids.append(151643)
                if max(ids) >= 152576 or min(ids) < 0:
                    raise ValueError("Out-of-vocabulary calibration token")
                offset = handles[split].tell() // 4
                np.asarray(ids, dtype=np.uint32).tofile(handles[split])
                entry = {
                    "id": identity,
                    "category": category,
                    "split": split,
                    "offset": offset,
                    "tokens": len(ids),
                }
                raw.write(
                    json.dumps({**entry, "text": text}, ensure_ascii=False) + "\n"
                )
                manifest.write(json.dumps(entry) + "\n")
                counts[f"{split}/{category}"] += len(ids)
                if counts[f"train/{category}"] >= budget:
                    break
            if counts[f"train/{category}"] < budget:
                raise RuntimeError(f"Source exhausted before {category} budget")
            status = {
                "stage": "tokenizing",
                "category": category,
                "tokens": dict(counts),
                "documents": len(seen),
            }
            (root / "status.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
        raw.close()
        manifest.close()
    for split in handles:
        data = np.memmap(root / f"{split}.uint32", mode="r", dtype=np.uint32)
        output = np.lib.format.open_memmap(
            root / f"{split}.npy", mode="w+", dtype=np.uint32, shape=data.shape
        )
        output[:] = data
        output.flush()
        if split != "train" and data.size < 16384:
            raise RuntimeError("Insufficient held-out data")
    result = {
        "complete": True,
        "tokens": dict(counts),
        "documents": len(seen),
        "tokenizer_revision": "73875d00b30a89ef8cc353a0b60b0e9f9561952d",
        "split": "sha256 document-disjoint 95/2.5/2.5; exact deduplication",
        "scope": "text calibration; preserve unmodified multimodal weights",
        "mixture": "GLM v32 public generators retokenized natively; no GLM IDs",
    }
    (root / "complete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
