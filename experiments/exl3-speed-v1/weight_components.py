# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP4 codebook contracts, shared by fitting and native execution."""

import torch

LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
SHIFTS = {2: (0, -4), 4: (0, -3, -6, -9)}


def validate_lut(lut, components=2, experts=None):
    """Return the per-expert byte stride; zero denotes one shared table."""
    if components not in SHIFTS:
        raise ValueError("weight_components must be 2 or 4")
    if lut.dtype != torch.uint8 or not lut.is_contiguous():
        raise ValueError("LUT must be contiguous uint8")
    pairs = components // 2
    if lut.shape == (pairs, 1024) or (components == 2 and lut.shape == (1024,)):
        return 0
    if experts is not None and (
        lut.shape == (experts, pairs, 1024)
        or (components == 2 and lut.shape == (experts, 1024))
    ):
        return pairs * 1024
    raise ValueError("Expected shared [pairs,1024] or expert [E,pairs,1024] LUT")


def expert_luts(lut, components, experts):
    """Normalize tables for concatenating gate and up physical experts."""
    stride = validate_lut(lut, components, experts)
    pairs = components // 2
    if stride:
        return lut.reshape(experts, pairs, 1024)
    return lut.reshape(1, pairs, 1024).expand(experts, -1, -1).contiguous()


def decode_lut(lut, components=2):
    """Decode component-major packed pairs into scalar codebook values."""
    pairs = components // 2
    if components not in SHIFTS or lut.shape[-1] != 1024:
        raise ValueError("Invalid codebook shape or component count")
    if lut.ndim == 1:
        lut = lut.unsqueeze(0)
    if lut.shape[-2] != pairs:
        raise ValueError("Component axis must contain components/2 packed pairs")
    levels = torch.tensor(LEVELS, device=lut.device)
    result = torch.zeros_like(lut[..., 0, :], dtype=torch.float32)
    for pair in range(pairs):
        byte = lut[..., pair, :].long()
        result += levels[byte & 15] * 2.0 ** SHIFTS[components][pair * 2]
        result += levels[byte >> 4] * 2.0 ** SHIFTS[components][pair * 2 + 1]
    return result


def make_four_lut(device):
    """Reference FP4 approximation; direct fitting may supply different tables."""
    from trellis_gemm import make_lut

    levels = torch.tensor(LEVELS, device=device)
    indices = torch.arange(65536, device=device)
    digits = torch.stack([(indices >> (4 * j)) & 15 for j in range(4)], 1)
    values = sum(levels[digits[:, j]] * 2.0**s for j, s in enumerate(SHIFTS[4]))
    keep = values == values.half().float()
    values, digits = values[keep], digits[keep]
    order = values.argsort(stable=True)
    values, digits = values[order], digits[order]
    target = make_lut(device).float()
    right = torch.searchsorted(values, target).clamp(max=len(values) - 1)
    left = (right - 1).clamp_min(0)
    chosen = torch.where(
        (target - values[left]).abs() <= (target - values[right]).abs(), left, right
    )
    digits = digits[chosen].byte()
    return (digits[:, ::2] | (digits[:, 1::2] << 4)).T.contiguous()
