# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked Python entry point for the native SM120 research kernel."""

import ctypes
from pathlib import Path

import torch
from trellis_gemm import reduce, reduce_had_scatter

_lib = ctypes.CDLL(str(Path(__file__).parent / "local-results/native_fp4.so"))
_lib.exl3_pack.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
_lib.exl3_project.argtypes = (
    [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
)


_lib.exl3_had_pack.argtypes = (
    [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
)


def pack(x, input_scales=None):
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float16
    m, k = x.shape
    assert m > 0 and k % 64 == 0
    q = torch.empty((m, 2, k // 8), device=x.device, dtype=torch.int32)
    scales = torch.empty((m, 2, k // 32), device=x.device, dtype=torch.uint8)
    with torch.accelerator.device_index(x.device.index):
        if input_scales is None:
            code = _lib.exl3_pack(
                x.data_ptr(),
                q.data_ptr(),
                scales.data_ptr(),
                m,
                k,
                torch.cuda.current_stream().cuda_stream,
            )
        else:
            assert (
                k % 128 == 0
                and input_scales.shape == (k,)
                and input_scales.dtype == torch.float16
            )
            assert input_scales.device == x.device and input_scales.is_contiguous()
            code = _lib.exl3_had_pack(
                x.data_ptr(),
                input_scales.data_ptr(),
                q.data_ptr(),
                scales.data_ptr(),
                m,
                k,
                torch.cuda.current_stream().cuda_stream,
            )
    if code:
        raise RuntimeError(f"exl3_pack CUDA error {code}")
    return q, scales


def project(x, packed, lut, splits=8, rows=None, scales=None, input_scales=None):
    assert packed.is_contiguous() and packed.dtype == torch.int16
    assert lut.is_contiguous() and lut.dtype == torch.uint8 and lut.numel() == 1024
    assert x.device == packed.device == lut.device
    m, k = x.shape
    assert packed.shape[0] * 16 == k and packed.shape[2] in (24, 32, 40)
    n = packed.shape[1] * 16
    assert 1 <= splits <= k // 64
    q, xs = pack(x, input_scales)
    scratch = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    with torch.accelerator.device_index(x.device.index):
        code = _lib.exl3_project(
            packed.data_ptr(),
            lut.data_ptr(),
            q.data_ptr(),
            xs.data_ptr(),
            scratch.data_ptr(),
            m,
            n,
            k,
            splits,
            packed.shape[2] // 8,
            torch.cuda.current_stream().cuda_stream,
        )
    if code:
        raise RuntimeError(f"exl3_project CUDA error {code}")
    if rows is None:
        reduce[((m * n + 255) // 256,)](scratch, out, m, n, splits, 256, num_warps=4)
    else:
        assert n % 128 == 0 and rows.shape == scales.shape == (n,)
        assert rows.device == scales.device == x.device
        reduce_had_scatter[(m, n // 128)](
            scratch, out, rows, scales, m, n, n, splits, num_warps=4
        )
    return out
