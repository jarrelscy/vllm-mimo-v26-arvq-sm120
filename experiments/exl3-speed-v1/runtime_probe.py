# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200 single-expert native EXL3 runtime versus dense BF16 reference."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from common import (
    ARTIFACT_ROOT,
    CANDIDATES,
    RESULT_ROOT,
    WORK_ROOT,
    Source,
    forward,
    load_data,
)
from decode import decode, load
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from trellis_gemm import (
    make_lut,
    make_pair_lut,
    multiply,
    multiply_scatter,
    reduce_had_scatter,
    swiglu,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument(
        "--backend",
        choices=[
            "official",
            "fused",
            "fp4",
            "fp4_residual",
            "fused_scatter",
            "fp4_scatter",
            "recon_scatter",
        ],
        default="official",
    )
    ap.add_argument("--splits", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer, expert, subdir, codec = CANDIDATES[args.index]
    path = ARTIFACT_ROOT / subdir / "selected.bin"
    values = load(path)
    weights = list(decode(path).values())
    data = load_data(WORK_ROOT, layer, expert)
    source = Source(Path("/data/jarrel/mimo-v26-arvq-hot5/source"))
    native = [
        source.expert(layer, expert, p, "cuda")
        for p in ("gate_proj", "up_proj", "down_proj")
    ]
    lut = make_pair_lut("cuda") if args.backend.startswith("fp4") else make_lut("cuda")
    groups = []
    for j in range(3):
        group = []
        offset = 0
        for v in values:
            if int(v["projection"]) != j:
                continue
            tensors = {
                k: t
                for k, t in v.items()
                if isinstance(t, torch.Tensor)
                and k not in ("projection", "blocks", "full_rows")
            }
            obj = LinearEXL3(None, v["shape"][1], v["shape"][0], **tensors)
            obj.config.infer_params.no_reconstruct = True
            rows = (
                (
                    v["blocks"].long()[:, None] * 128 + torch.arange(128, device="cuda")
                ).flatten()
                if "blocks" in v
                else None
            )
            if args.backend.endswith("scatter") and rows is None:
                rows = torch.arange(offset, offset + v["shape"][0], device="cuda")
            offset += v["shape"][0]
            group.append((obj, rows))
        groups.append(group)

    def project(x, j, reconstruct):
        if args.backend.endswith("scatter") and not reconstruct:
            output = torch.empty(
                len(x), weights[j].shape[0], device="cuda", dtype=torch.float32
            )
            for obj, rows in groups[j]:
                xh = torch.empty_like(x)
                ext.had_r_128(x, xh, obj.suh, None, 1.0)
                if args.backend == "recon_scatter":
                    w = obj.get_inner_weight_tensor()
                    y = torch.empty(
                        (len(x), obj.out_features), device="cuda", dtype=torch.float32
                    )
                    ext.hgemm_recon(xh, w, y)
                    reduce_had_scatter[(len(x), obj.out_features // 128)](
                        y,
                        output,
                        rows,
                        obj.svh,
                        len(x),
                        obj.out_features,
                        output.shape[1],
                        1,
                        num_warps=4,
                    )
                    continue
                multiply_scatter(
                    xh,
                    obj.trellis,
                    lut,
                    output,
                    rows,
                    obj.svh,
                    args.splits,
                    fp4=args.backend.startswith("fp4"),
                    residual=args.backend.startswith("fp4"),
                )
            return output
        chunks = []
        for obj, _ in groups[j]:
            if args.backend != "official" and not reconstruct:
                xh = torch.empty_like(x)
                ext.had_r_128(x, xh, obj.suh, None, 1.0)
                y = multiply(
                    xh,
                    obj.trellis,
                    lut,
                    args.splits,
                    fp4=args.backend.startswith("fp4"),
                    residual=args.backend == "fp4_residual",
                )
                ext.had_r_128(y, y, None, obj.svh, 1.0)
                chunks.append(y)
            else:
                chunks.append(
                    obj.forward(
                        x, {"reconstruct": reconstruct}, out_dtype=torch.float32
                    )
                )
        if groups[j][0][1] is None:
            return torch.cat(chunks, dim=-1)
        result = torch.empty(len(x), weights[j].shape[0], device="cuda")
        for (_, rows), chunk in zip(groups[j], chunks):
            result.index_copy_(1, rows, chunk)
        return result

    def run(x, reconstruct=False):
        gate = project(x, 0, reconstruct)
        up = project(x, 1, reconstruct)
        hidden = (
            swiglu(gate, up)
            if args.backend.endswith("scatter")
            else (F.silu(gate) * up).half().contiguous()
        )
        return project(hidden, 2, reconstruct)

    def timing(fn):
        for _ in range(3):
            fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        times = []
        for _ in range(5):
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 10.0)
        return sorted(times)[2]

    report = {
        "layer": layer,
        "expert": expert,
        "backend": args.backend,
        "splits": args.splits,
        "block_m": int(os.environ.get("MIMO_BM", 16)),
        "block_n": int(os.environ.get("MIMO_BN", 64)),
        "gpu": torch.cuda.get_device_name(),
        "scope": (
            "single-expert CUDA-graph latency; excludes routing and "
            "cross-GPU communication; dense baseline stores expanded "
            "weights"
        ),
        "timings": [],
        "quality": {},
    }
    with torch.no_grad():
        for split, name in ((1, "validation"), (2, "audit")):
            x = data[split]["x"]
            p = data[split]["p"][:, None]
            target = forward(x, native)
            pred = torch.cat([run(part.half().contiguous()) for part in x.split(64)])
            report["quality"][name] = float(
                ((pred - target) * p).double().norm() / (target * p).double().norm()
            )
        dense = [w.bfloat16() for w in weights]
        for batch in (1, 4, 16, 64, 256):
            x = (
                data[1]["x"]
                .repeat(((batch + len(data[1]["x"]) - 1) // len(data[1]["x"]), 1))[
                    :batch
                ]
                .half()
                .contiguous()
            )
            if len(x) != batch:
                continue
            row = {
                "batch": batch,
                "compressed_path_us": timing(lambda x=x: run(x)),
                "reconstruct_each_call_us": timing(lambda x=x: run(x, True)),
                "cached_dense_bf16_us": timing(lambda x=x: forward(x, dense)),
            }
            report["timings"].append(row)
            print(row, flush=True)
    out = RESULT_ROOT
    out.mkdir(exist_ok=True)
    (
        out
        / (
            f"runtime_{args.backend}_s{args.splits}_"
            f"bm{report['block_m']}_bn{report['block_n']}_"
            f"layer{layer}_expert{expert}.json"
        )
    ).write_text(json.dumps(report, indent=2) + "\n")
    print(report["quality"], flush=True)


if __name__ == "__main__":
    main()
