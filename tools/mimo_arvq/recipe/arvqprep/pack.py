"""Pure-torch ARVQ serialization: native MMA-fragment index packing + codebook
and scale layout, byte-compatible with the branch's CUDA tools.

Ports (no CUDA toolchain needed; SM120 nvcc unavailable on the A100 fit box):
  - examples/arvq/convert/assign.cu::pack_kernel / pair_at   -> pack_indices
  - examples/arvq/convert/build_checkpoint.py::check_indices -> decode_indices
  - examples/arvq/convert/format_utils.py::pack_cb           -> pack_codebooks
  - examples/arvq/convert/build_checkpoint.py scale reshape  -> pack_scales

The packed stream is [N/16, K/64, 60] uint32 per expert (no guard word on disk;
the vLLM loader appends one at load time). pack_indices <-> decode_indices is an
exact round-trip and is asserted in the smoke test.
"""
from __future__ import annotations

import torch

from . import LEVELS


def _pos_maps(device):
    """pos in [0,128) -> (row-within-tile in [0,16), col-within-64-group in [0,8))."""
    pos = torch.arange(128, device=device)
    j = pos // 32
    lane = pos % 32
    q = lane // 4
    c = lane % 4
    row_in_tile = q + 8 * (j & 1)          # [128] in [0,16)
    kg_in_g = (j // 2) * 4 + c             # [128] in [0,8)
    return row_in_tile, kg_in_g


def pack_indices(a: torch.Tensor, b: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """a,b uint8 [N, K/8] natural indices -> uint32 [N/16, K/64, 60] fragment stream.

    Mirrors assign.cu pack_kernel: 128 15-bit pairs per (16-row tile, 64-col
    group) packed LSB-first into 60 uint32 words.
    """
    assert N % 16 == 0 and K % 64 == 0
    assert a.shape == b.shape == (N, K // 8)
    device = a.device
    pair = a.long() | (b.long() << 8)                      # [N, K/8], 15-bit
    row_in_tile, kg_in_g = _pos_maps(device)               # [128], [128]
    nt, ng = N // 16, K // 64
    rows = (torch.arange(nt, device=device)[:, None, None] * 16
            + row_in_tile[None, None, :])                  # [nt,1,128]
    kgs = (torch.arange(ng, device=device)[None, :, None] * 8
           + kg_in_g[None, None, :])                       # [1,ng,128]
    ordered = pair[rows, kgs]                              # [nt, ng, 128]
    out = torch.zeros(nt, ng, 60, dtype=torch.int64, device=device)
    for w in range(60):
        bit = w * 32
        pos0 = bit // 15
        shift = bit % 15
        v = ordered[..., pos0].clone()
        if pos0 + 1 < 128:
            v |= ordered[..., pos0 + 1] << 15
        if pos0 + 2 < 128:
            v |= ordered[..., pos0 + 2] << 30
        if pos0 + 3 < 128:
            v |= ordered[..., pos0 + 3] << 45
        out[..., w] = (v >> shift) & 0xFFFFFFFF
    return out.to(torch.uint32)


def decode_indices(packed: torch.Tensor, N: int, K: int):
    """uint32 [N/16, K/64, 60] -> (a,b) uint8 [N,K/8]. Independent bit decode
    (mirrors build_checkpoint.check_indices) for round-trip verification."""
    assert packed.shape == (N // 16, K // 64, 60)
    device = packed.device
    words = packed.long()
    row_in_tile, kg_in_g = _pos_maps(device)
    nt, ng = N // 16, K // 64
    pos = torch.arange(128, device=device)
    bit = pos * 15
    word = bit // 32
    shift = bit % 32
    w0 = words[:, :, word]                                 # [nt,ng,128]
    v = w0 >> shift[None, None, :]
    need = shift > 17
    if need.any():
        w1 = words[:, :, (word + 1).clamp(max=59)]
        v = torch.where(need[None, None, :],
                        v | (w1 << (32 - shift)[None, None, :]), v)
    pair = (v & 32767)
    a_ord = (pair & 255).to(torch.uint8)                  # [nt,ng,128]
    b_ord = ((pair >> 8) & 127).to(torch.uint8)
    a = torch.empty(N, K // 8, dtype=torch.uint8, device=device)
    b = torch.empty(N, K // 8, dtype=torch.uint8, device=device)
    rows = (torch.arange(nt, device=device)[:, None, None] * 16
            + row_in_tile[None, None, :]).expand(nt, ng, 128)
    kgs = (torch.arange(ng, device=device)[None, :, None] * 8
           + kg_in_g[None, None, :]).expand(nt, ng, 128)
    a[rows.reshape(-1), kgs.reshape(-1)] = a_ord.reshape(-1)
    b[rows.reshape(-1), kgs.reshape(-1)] = b_ord.reshape(-1)
    return a, b


def pack_codebooks(c0: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
    """c0[256,8], c1[128,8] FP4-on-grid floats -> uint32 [384] (8 nibbles each).

    Mirrors format_utils.pack_cb exactly (nibble = argmin |value - LEVELS|)."""
    c = torch.cat([c0, c1]).to(torch.float32)
    levels = torch.tensor(LEVELS, device=c.device, dtype=torch.float32)
    n = (c[:, :, None] - levels).abs().argmin(-1)          # [384,8] in [0,16)
    assert torch.equal(levels[n], c), "codebook entries not on the FP4 grid"
    shifts = (torch.arange(8, device=c.device) * 4)
    return (n.to(torch.int64) << shifts).sum(-1).to(torch.int32).view(torch.uint32)


def pack_scales(s_uint8: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Per-(row,128block) E4M3 scale bytes [E,N,K/128] uint8 ->
    [E,N/16,K/128,16] uint8 (tile-major, 16 rows last), matching the loader."""
    E = s_uint8.shape[0]
    assert s_uint8.shape == (E, N, K // 128)
    return (s_uint8.reshape(E, N // 16, 16, K // 128)
            .permute(0, 1, 3, 2).contiguous())


def levels_tensor(device):
    return torch.tensor(LEVELS, device=device, dtype=torch.float32)


def project_to_fp4(c: torch.Tensor) -> torch.Tensor:
    """Snap each entry to the nearest FP4 level (mirrors fit_codebooks.project)."""
    levels = levels_tensor(c.device)
    return levels[(c[..., None] - levels).abs().argmin(-1)]
