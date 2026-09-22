# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 hot weights and frozen-output arithmetic for the hybrid campaign."""

import json
from pathlib import Path

import torch
from propagation_math import expert_from_packed_input
from safetensors.torch import load_file, save_file


def quantize(weight):
    """E2M1 weights, E4M3 scale per 16 and a FP32 tensor scale."""
    weight = weight.float()
    global_scale = (weight.abs().max() / (6 * 448)).clamp_min(1e-20)
    groups = weight.reshape(*weight.shape[:-1], -1, 16)
    scales = (groups.abs().amax(-1) / (6 * global_scale)).clamp(0, 448)
    scales = scales.to(torch.float8_e4m3fn).float()
    normalized = groups / (global_scale * scales[..., None]).clamp_min(1e-30)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    codes = (normalized.abs()[..., None] - levels).abs().argmin(-1)
    codes = (codes | ((normalized < 0).long() << 3)).byte().reshape(weight.shape)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    return packed, scales.to(torch.float8_e4m3fn).view(torch.uint8), global_scale


def decode(packed, scales, global_scale):
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    scale = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return levels[codes] * scale * global_scale


def quantized_expert(source, layer, expert, device):
    values = [
        quantize(source.expert(layer, expert, p, device))
        for p in ("gate_proj", "up_proj", "down_proj")
    ]
    return values, (torch.cat([decode(*v) for v in values[:2]]), decode(*values[2]))


def allocation(work, layer):
    path = Path(work) / "allocation.json"
    if not path.exists():
        return [], list(range(384))
    hot = json.loads(path.read_text())["layers"][str(layer)]["hot"]
    return hot, [e for e in range(384) if e not in hot]


def export_hot(source, layer, hot, path, device):
    prefix = f"model.layers.{layer}.mlp.experts."
    kinds = torch.full((384,), 2, dtype=torch.int8)
    kinds[hot] = 0
    tensors = {prefix + "hyb_kind": kinds}
    collected = {
        k: []
        for k in (
            "w13_packed",
            "w13_bscale",
            "w13_scale2",
            "w2_packed",
            "w2_bscale",
            "w2_scale2",
        )
    }
    for expert in hot:
        values, _ = quantized_expert(source, layer, expert, device)
        g, u, d = values
        for name, value in {
            "w13_packed": torch.cat((g[0], u[0])),
            "w13_bscale": torch.cat((g[1], u[1])),
            "w13_scale2": torch.stack((g[2], u[2])),
            "w2_packed": d[0],
            "w2_bscale": d[1],
            "w2_scale2": d[2][None],
        }.items():
            collected[name].append(value.cpu())
    if hot:
        tensors.update(
            {prefix + "nvfp4_" + k: torch.stack(v) for k, v in collected.items()}
        )
    else:
        for name, shape in {
            "w13_packed": (0, 4096, 3072),
            "w13_bscale": (0, 4096, 384),
            "w13_scale2": (0, 2),
            "w2_packed": (0, 6144, 1024),
            "w2_bscale": (0, 6144, 128),
            "w2_scale2": (0, 1),
        }.items():
            tensors[prefix + "nvfp4_" + name] = torch.empty(
                shape, dtype=torch.float32 if name.endswith("scale2") else torch.uint8
            )
    save_file(tensors, str(path))


def load_hot(work, layer, device):
    hot, _ = allocation(work, layer)
    if not hot:
        return {}
    tensors = load_file(
        str(Path(work) / "hot" / f"layer{layer}.safetensors"), device=device
    )
    prefix = f"model.layers.{layer}.mlp.experts.nvfp4_"
    decoded = {}
    for slot, expert in enumerate(hot):
        packed = tensors[prefix + "w13_packed"][slot]
        scales = tensors[prefix + "w13_bscale"][slot]
        globals_ = tensors[prefix + "w13_scale2"][slot]
        gate, up = packed.chunk(2)
        sg, su = scales.chunk(2)
        w13 = torch.cat((decode(gate, sg, globals_[0]), decode(up, su, globals_[1])))
        w2 = decode(
            tensors[prefix + "w2_packed"][slot],
            tensors[prefix + "w2_bscale"][slot],
            tensors[prefix + "w2_scale2"][slot],
        )
        decoded[expert] = (w13, w2)
    return decoded


def hot_output(packed_input, ids, gates, weights):
    output = torch.zeros_like(packed_input)
    for expert, (w13, w2) in weights.items():
        rows, slots = torch.where(ids == expert)
        if len(rows):
            value = expert_from_packed_input(packed_input[rows], w13, w2)
            output.index_add_(0, rows, value * gates[rows, slots, None])
    return output
