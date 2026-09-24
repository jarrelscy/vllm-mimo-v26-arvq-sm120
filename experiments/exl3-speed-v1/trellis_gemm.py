# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental fused trellis decode and FP16 tensor-core GEMM for B200."""

import os

import torch
import triton as tr
import triton.language as tl


@tr.jit
def decode_symbols(
    P, LUT, kk, nn, N: tl.constexpr, KA: tl.constexpr, HALF: tl.constexpr
):
    r = kk % 16
    c = nn % 16
    lane = (c % 8) * 4 + (r % 8) // 2
    element = r % 2 + 2 * (r // 8) + 4 * (c // 8)
    position = lane * 8 + element
    total: tl.constexpr = 256 * KA + 128 * HALF
    words: tl.constexpr = total // 32
    end = (position + 1) * KA + ((position + 1) // 2) * HALF + total
    begin = end - 16
    i0 = (begin // 32) % words
    i1 = ((end - 1) // 32) % words
    shift = (32 - end % 32) % 32
    tile = (kk // 16 * (N // 16) + nn // 16) * words
    a = tl.load(P + tile + i0).to(tl.uint32)
    b = tl.load(P + tile + i1).to(tl.uint32)
    merged = (a.to(tl.uint64) << 32) | b.to(tl.uint64)
    state = ((merged >> shift) & 65535).to(tl.uint32)
    product = state * 0x83DCD12D
    byte_sum = (
        (product & 255)
        + ((product >> 8) & 255)
        + ((product >> 16) & 255)
        + (product >> 24)
    )
    return tl.load(LUT + byte_sum).to(tl.float16)


@tr.jit
def gemm(
    X,
    P,
    LUT,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    KA: tl.constexpr,
    HALF: tl.constexpr,
    SPLITS: tl.constexpr,
    RESIDUAL: tl.constexpr = False,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 128,
):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    ps = tl.program_id(2)
    mm = pm * BM + tl.arange(0, BM)
    nn = pn * BN + tl.arange(0, BN)
    kr = tl.arange(0, BK)
    steps: tl.constexpr = tr.cdiv(K, BK * SPLITS)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(steps):
        kk = (ps * steps + step) * BK + kr
        safe = tl.minimum(kk, K - 1)
        weight = decode_symbols(P, LUT, safe[:, None], nn[None, :], N, KA, HALF)
        xx = tl.load(
            X + mm[:, None] * K + kk[None, :], (mm[:, None] < M) & (kk[None, :] < K), 0
        )
        acc += tl.dot(xx, weight)
    tl.store(Y + ps * M * N + mm[:, None] * N + nn[None, :], acc, mm[:, None] < M)


@tr.jit
def gemm_fp4(
    X,
    P,
    LUT,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    KA: tl.constexpr,
    HALF: tl.constexpr,
    SPLITS: tl.constexpr,
    RESIDUAL: tl.constexpr = False,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 128,
):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    ps = tl.program_id(2)
    mm = pm * BM + tl.arange(0, BM)
    nn = pn * BN + tl.arange(0, BN)
    kr = tl.arange(0, BK)
    steps: tl.constexpr = tr.cdiv(K, BK * SPLITS)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(steps):
        kk = (ps * steps + step) * BK + kr
        tl.minimum(kk, K - 1)
        xx = tl.load(
            X + mm[:, None] * K + kk[None, :], (mm[:, None] < M) & (kk[None, :] < K), 0
        )
        kk2 = (ps * steps + step) * BK + tl.arange(0, BK // 2) * 2
        pair0 = decode_symbols(
            P, LUT, tl.minimum(kk2, K - 1)[:, None], nn[None, :], N, KA, HALF
        ).to(tl.uint8)
        pair1 = decode_symbols(
            P, LUT, tl.minimum(kk2 + 1, K - 1)[:, None], nn[None, :], N, KA, HALF
        ).to(tl.uint8)
        plane0 = (pair0 & 15) | ((pair1 & 15) << 4)
        plane1 = (pair0 >> 4) | (pair1 & 240)
        grouped = tl.reshape(xx.to(tl.float32), (BM, BK // 32, 32))
        peak = tl.max(tl.abs(grouped), axis=2)
        exponent = tl.ceil(tl.log2(tl.maximum(peak / 448.0, 1.0e-12)))
        scale = tl.exp2(exponent)
        normalized = tl.reshape(grouped / scale[:, :, None], (BM, BK)).to(tl.float8e4nv)
        sx = (exponent + 127).to(tl.uint8)
        sw0 = tl.full((BN, BK // 32), 127, tl.uint8)
        sw1 = tl.full((BN, BK // 32), 123, tl.uint8)
        acc = tl.dot_scaled(normalized, sx, "e4m3", plane0, sw0, "e2m1", acc)
        acc = tl.dot_scaled(normalized, sx, "e4m3", plane1, sw1, "e2m1", acc)
        if RESIDUAL:
            rem = (
                tl.reshape(grouped / scale[:, :, None], (BM, BK))
                - normalized.to(tl.float32)
            ) * 16.0
            rem = rem.to(tl.float8e4nv)
            sr = (exponent + 123).to(tl.uint8)
            acc = tl.dot_scaled(rem, sr, "e4m3", plane0, sw0, "e2m1", acc)
            acc = tl.dot_scaled(rem, sr, "e4m3", plane1, sw1, "e2m1", acc)
    tl.store(Y + ps * M * N + mm[:, None] * N + nn[None, :], acc, mm[:, None] < M)


@tr.jit
def reduce(Y, OUT, M: tl.constexpr, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ss = tl.arange(0, tr.next_power_of_2(S))
    vals = tl.load(
        Y + ss[:, None] * M * N + i[None, :],
        (ss[:, None] < S) & (i[None, :] < M * N),
        0,
    )
    tl.store(OUT + i, tl.sum(vals, axis=0), i < M * N)


def make_lut(device):
    sums = torch.arange(1024, device=device, dtype=torch.int32)
    values = (sums + 0x6400).short().view(torch.float16).float()
    scale = (
        torch.tensor([0x1EEE], device=device, dtype=torch.int16)
        .view(torch.float16)
        .float()
    )
    bias = (
        torch.tensor([-14031], device=device, dtype=torch.int16)
        .view(torch.float16)
        .float()
    )
    return (values * scale + bias).half()


def multiply(x, packed, lut, splits=16, fp4=False, residual=False):
    m, k = x.shape
    n = packed.shape[1] * 16
    rate = packed.shape[-1] / 16
    ka = int(rate)
    half = int(rate != ka)
    scratch = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    kernel = gemm_fp4 if fp4 else gemm
    bm = int(os.environ.get("MIMO_BM", "16"))
    bn = int(os.environ.get("MIMO_BN", "64"))
    kernel[(tr.cdiv(n, bn), tr.cdiv(m, bm), splits)](
        x,
        packed.view(torch.int32),
        lut,
        scratch,
        m,
        n,
        k,
        ka,
        half,
        splits,
        residual,
        BM=bm,
        BN=bn,
        num_warps=4,
    )
    reduce[(tr.cdiv(m * n, 256),)](scratch, output, m, n, splits, 256, num_warps=4)
    return output


def make_pair_lut(device):
    exact = make_lut(device).float()
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=device,
    )
    values = (levels[:, None] + levels[None, :] / 16).flatten()
    idx = (exact[:, None] - values[None, :]).abs().argmin(1)
    return ((idx // 16) | ((idx % 16) << 4)).byte()


@tr.jit
def reduce_had_scatter(
    Y,
    OUT,
    Rows,
    SV,
    M: tl.constexpr,
    N: tl.constexpr,
    FULL_N: tl.constexpr,
    S: tl.constexpr,
):
    m = tl.program_id(0)
    block = tl.program_id(1)
    ii = tl.arange(0, 128)
    ss = tl.arange(0, tr.next_power_of_2(S))
    vals = tl.load(
        Y + ss[:, None] * M * N + m * N + block * 128 + ii[None, :], ss[:, None] < S, 0
    )
    v = tl.sum(vals, axis=0)
    for stage in tl.static_range(7):
        stride = 1 << stage
        other = tl.gather(v, ii ^ stride, axis=0)
        v = tl.where((ii & stride) == 0, v + other, other - v)
    v = v * 0.08838834764831845 * tl.load(SV + block * 128 + ii).to(tl.float32)
    rows = tl.load(Rows + block * 128 + ii)
    tl.store(OUT + m * FULL_N + rows, v)


@tr.jit
def swiglu_kernel(G, U, H, COUNT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(G + i, i < COUNT, 0)
    u = tl.load(U + i, i < COUNT, 0)
    tl.store(H + i, g / (1 + tl.exp(-g)) * u, i < COUNT)


def swiglu(g, u):
    h = torch.empty(g.shape, device=g.device, dtype=torch.float16)
    swiglu_kernel[(tr.cdiv(g.numel(), 256),)](g, u, h, g.numel(), 256)
    return h


def multiply_scatter(
    x, packed, lut, output, rows, sv, splits=8, fp4=False, residual=False
):
    m, k = x.shape
    n = packed.shape[1] * 16
    rate = packed.shape[-1] / 16
    ka = int(rate)
    half = int(rate != ka)
    scratch = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    kernel = gemm_fp4 if fp4 else gemm
    bm = int(os.environ.get("MIMO_BM", "16"))
    bn = int(os.environ.get("MIMO_BN", "64"))
    kernel[(tr.cdiv(n, bn), tr.cdiv(m, bm), splits)](
        x,
        packed.view(torch.int32),
        lut,
        scratch,
        m,
        n,
        k,
        ka,
        half,
        splits,
        residual,
        BM=bm,
        BN=bn,
        num_warps=4,
    )
    reduce_had_scatter[(m, n // 128)](
        scratch, output, rows, sv, m, n, output.shape[1], splits, num_warps=4
    )
