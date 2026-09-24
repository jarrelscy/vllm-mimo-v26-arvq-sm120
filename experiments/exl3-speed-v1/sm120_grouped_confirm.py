# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Fixed-policy grouped MoE validation; GPU routing is included in graph timing."""

import argparse
import json
from pathlib import Path

import torch
from grouped_fp4 import GroupedMoE
from native_fp4 import project
from official_grouped import OfficialGrouped
from sm120_microbench import timing
from trellis_gemm import make_pair_lut

POLICY = {1: (8, 4), 2: (4, 1), 3: (8, 1), 4: (8, 1)}


def check_metadata(native, ids):
    flat = [e for row in ids for e in row]
    unique = list(dict.fromkeys(flat))
    slots = [i for e in unique for i, v in enumerate(flat) if e == v]
    r = len(flat)
    inverse = [slots.index(i) for i in range(r)]
    offsets = [0]
    for e in unique:
        offsets.append(offsets[-1] + flat.count(e))
    offsets += [r] * (r - len(unique))
    torch.testing.assert_close(
        native.inverse.cpu(), torch.tensor(inverse, dtype=torch.int32)
    )
    torch.testing.assert_close(
        native.offsets.cpu(),
        torch.tensor(offsets[:-1] + [r + x for x in offsets], dtype=torch.int32),
    )
    assert native.ids[: len(unique)].tolist() == unique
    assert native.ids[r : r + len(unique)].tolist() == [e + native.e for e in unique]
    assert native.gather[:r].tolist() == [i // native.topk for i in slots]


def reference(parts, lut, x, ids, weights, splits):
    h, inter = x.shape[1], parts[0][2].shape[1]
    ref = torch.zeros_like(x, dtype=torch.float32)
    for t, row in enumerate(ids):
        for j, e in enumerate(row):
            gu = [
                project(
                    x[t : t + 1],
                    p[e],
                    lut,
                    splits=splits[0],
                    rows=torch.arange(inter, device=x.device),
                    scales=sv[e],
                    input_scales=su[e],
                    arvq=True,
                    gemv=True,
                )
                for p, su, sv in parts[:2]
            ]
            activation = (torch.nn.functional.silu(gu[0]) * gu[1]).half()
            p, su, sv = parts[2]
            y = project(
                activation,
                p[e],
                lut,
                splits=splits[1],
                rows=torch.arange(h, device=x.device),
                scales=sv[e],
                input_scales=su[e],
                arvq=True,
                gemv=True,
            )
            ref[t : t + 1] += y * weights[t, j].float()
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=[1, 4], default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    torch.set_num_threads(4)
    e, h, inter = 32, 6144, 2048 // args.tp
    if args.small:
        h, inter = 256, 256
    lut = make_pair_lut("cuda")
    report = {
        "tp": args.tp,
        "policy": POLICY,
        "gpu": torch.cuda.get_device_name(),
        "rows": [],
    }
    for seed in range(args.seeds):
        torch.manual_seed(91387 + seed)
        parts = []
        for k, n in [(h, inter), (h, inter), (inter, h)]:
            p = torch.randint(
                -32768,
                32767,
                (e, k // 16, n // 16, 32),
                device="cuda",
                dtype=torch.int16,
            )
            # Signed, independent per-expert Hadamard scales.
            su = torch.randn(e, k, device="cuda", dtype=torch.float16)
            sv = torch.randn(e, n, device="cuda", dtype=torch.float16)
            parts.append((p, su, sv))
        official = None if args.parity_only else OfficialGrouped(*parts)
        for m in [args.batch] if args.batch else range(1, 5):
            splits = tuple(min(s, k // 64) for s, k in zip(POLICY[m], [h, inter]))
            native = GroupedMoE(
                *parts, lut, m, splits_gu=splits[0], splits_down=splits[1]
            )
            x = torch.randn(m, h, device="cuda", dtype=torch.float16) * 0.01
            weights = torch.rand(m, 8, device="cuda").softmax(-1).half()
            selected = torch.empty(m, 8, device="cuda", dtype=torch.int64)
            graph = torch.cuda.CUDAGraph()
            native(x, selected.zero_().add_(torch.arange(8, device="cuda")), weights)
            torch.accelerator.synchronize()
            with torch.cuda.graph(graph):
                native(x, selected, weights)
            for pattern in ["shared", "pairs", "disjoint", "mixed"]:
                perm = torch.randperm(e).tolist()
                ids = []
                for t in range(m):
                    group = (
                        0
                        if pattern == "shared"
                        else (t // 2 if pattern == "pairs" else t)
                    )
                    row = list(range(group * 8, group * 8 + 8))
                    if pattern == "mixed":
                        row = list(range(4)) + list(range(4 + 4 * t, 8 + 4 * t))
                    ids.append([perm[i] for i in row])
                selected.copy_(torch.tensor(ids, device="cuda"))
                eager = native(x, selected, weights).clone()
                graph.replay()
                out = native.output.clone()
                torch.testing.assert_close(out, eager, rtol=0, atol=0)
                check_metadata(native, ids)
                ref = reference(parts, lut, x, ids, weights, splits)
                rel = ((out - ref).norm() / ref.norm()).item()
                max_abs = (out - ref).abs().max().item()
                assert torch.isfinite(out).all() and rel < 0.0002, (m, pattern, rel)
                row = dict(
                    seed=seed, m=m, pattern=pattern, relative_l2=rel, max_abs=max_abs
                )
                if not args.parity_only:
                    calls = [
                        ("native_us", lambda: native(x, selected, weights)),
                        ("official_us", lambda: official(x, selected, weights)),
                    ]
                    if (seed + m) % 2:
                        calls.reverse()
                    for key, fn in calls:
                        row[key] = timing(fn)
                    row["speedup"] = row["official_us"] / row["native_us"]
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
            native = graph = None
        parts = official = p = su = sv = calls = fn = None
        torch.accelerator.empty_cache()
    path = Path(
        f"local-results/grouped_confirm_tp{args.tp}"
        f"{'_small' if args.small else ''}"
        f"{'_parity' if args.parity_only else ''}"
        f"{'_m' + str(args.batch) if args.batch else ''}.json"
    )
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
