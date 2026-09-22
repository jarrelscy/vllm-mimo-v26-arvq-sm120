# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-streamed reference math for the released MiMo weight layout."""

import torch
import torch.nn.functional as F
from source import Source


def rms(x, weight, eps=1e-5):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    ) * weight


def rotary(x, theta):
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    frequency = theta ** (-torch.arange(0, 64, 2, device=x.device).float() / 64)
    angles = positions[:, None] * frequency[None, :]
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    rotated, rest = x[..., :64], x[..., 64:]
    first, second = rotated.chunk(2, dim=-1)
    return torch.cat((rotated * cos + torch.cat((-second, first), -1) * sin, rest), -1)


def attention(q, k, v, *, window=None, sink=None, eager=False):
    """Pad V for SDPA; an extra zero-value position implements learned sinks."""
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    length = q.shape[-2]
    positions = torch.arange(length, device=q.device)
    allowed = positions[:, None] >= positions[None, :]
    if window:
        allowed &= positions[:, None] - positions[None, :] < window
    mask = torch.zeros((length, length), device=q.device, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    mask = mask[None, None]
    if sink is not None:
        k = torch.cat((k, torch.zeros_like(k[..., :1, :])), dim=-2)
        v = torch.cat((v, torch.zeros_like(v[..., :1, :])), dim=-2)
        mask = torch.cat(
            (
                mask.expand(1, q.shape[1], length, length),
                sink.float().reshape(1, -1, 1, 1).expand(1, -1, length, 1),
            ),
            -1,
        )
    if eager:
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            q.shape[-1] ** -0.5
        )
        return torch.matmul((scores + mask).softmax(-1).to(v.dtype), v)
    dimension = v.shape[-1]
    padded = F.pad(v, (0, q.shape[-1] - dimension))
    return F.scaled_dot_product_attention(
        q, k, padded, attn_mask=mask.to(q.dtype), scale=q.shape[-1] ** -0.5
    )[..., :dimension]


class Layer:
    def __init__(self, source, index, device="cuda"):
        self.source = source if isinstance(source, Source) else Source(source)
        self.index, self.device = index, device
        self.config = self.source.config
        self.prefix = f"model.layers.{index}."
        self.window = 128 if self.config["hybrid_layer_pattern"][index] else None
        self.norm1 = self.weight("input_layernorm.weight")
        self.norm2 = self.weight("post_attention_layernorm.weight")
        raw = self.source.tensor(self.prefix + "self_attn.qkv_proj.weight", device)
        scales = self.source.tensor(
            self.prefix + "self_attn.qkv_proj.weight_scale_inv", device
        )
        # Source groups are [16 Q heads | one K head | one V head].
        # FP8 scale blocks restart at each group, including padded boundary rows.
        self.qkv = []
        for group in range(8):
            values = raw[group * 3392 : (group + 1) * 3392].float()
            scale = scales[group * 27 : (group + 1) * 27]
            expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:3392]
            self.qkv.append((values * expanded).to(torch.bfloat16))
        q, k, v = zip(*(w.split((3072, 192, 128), dim=0) for w in self.qkv))
        self.qkv = torch.cat((*q, *k, *v), dim=0)
        self.out = self.weight("self_attn.o_proj.weight")
        sink_key = self.prefix + "self_attn.attention_sink_bias"
        self.sink = (
            self.source.tensor(sink_key, device)
            if sink_key in self.source.index
            else None
        )
        if index:
            # Routing corrections are stored FP32. Rounding them changes top-k
            # near selection boundaries even when expert weights are unchanged.
            self.gate = self.source.tensor(self.prefix + "mlp.gate.weight", device)
            self.bias = self.source.tensor(
                self.prefix + "mlp.gate.e_score_correction_bias", device
            )
        else:
            self.dense = [
                self.weight(f"mlp.{p}.weight")
                for p in ("gate_proj", "up_proj", "down_proj")
            ]
        self.experts = None

    def weight(self, name):
        key = self.prefix + name
        weight = self.source.tensor(key, self.device)
        scale_key = key.removesuffix("weight") + "weight_scale_inv"
        if scale_key in self.source.index:
            scale = self.source.tensor(scale_key, self.device)
            weight = (
                weight.float()
                * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[
                    : weight.shape[0], : weight.shape[1]
                ]
            )
        return weight.to(torch.bfloat16)

    @torch.no_grad()
    def front(self, state, eager=False):
        x = rms(state, self.norm1)
        q, k, v = F.linear(x, self.qkv).split((24576, 1536, 1024), dim=-1)
        batch, length = state.shape[:2]
        q = q.reshape(batch, length, 128, 192).transpose(1, 2)
        k = k.reshape(batch, length, 8, 192).transpose(1, 2)
        v = v.reshape(batch, length, 8, 128).transpose(1, 2) * 0.612
        theta = 10000 if self.window else 10000000
        y = attention(
            rotary(q, theta),
            rotary(k, theta),
            v,
            window=self.window,
            sink=self.sink,
            eager=eager,
        )
        residual = state + F.linear(
            y.transpose(1, 2).reshape(batch, length, 16384), self.out
        )
        return residual, rms(residual, self.norm2)

    @torch.no_grad()
    def route(self, x):
        scores = F.linear(x.float(), self.gate.float()).sigmoid()
        ids = (scores + self.bias.float()).topk(8, dim=-1, sorted=False).indices
        weights = scores.gather(-1, ids)
        return ids, weights / (weights.sum(-1, keepdim=True) + 1e-20)

    def load_experts(self):
        if self.experts is None:
            self.experts = [
                tuple(
                    self.source.expert(self.index, e, p, self.device).to(torch.bfloat16)
                    for p in ("gate_proj", "up_proj", "down_proj")
                )
                for e in range(384)
            ]

    @torch.no_grad()
    def native_moe(self, x, ids, gates):
        self.load_experts()
        output = torch.zeros_like(x, dtype=torch.float32)
        for e, (wg, wu, wd) in enumerate(self.experts):
            rows, slots = torch.where(ids == e)
            if len(rows):
                z = x[rows]
                y = F.linear(F.silu(F.linear(z, wg)) * F.linear(z, wu), wd)
                output.index_add_(0, rows, y.float() * gates[rows, slots, None])
        return output

    @torch.no_grad()
    def dense_forward(self, state):
        residual, x = self.front(state)
        wg, wu, wd = self.dense
        return residual + F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)
