# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked Python entry point for the native SM120 research kernel."""

import ctypes
from pathlib import Path

import torch
from trellis_gemm import reduce, reduce_had_scatter
from weight_components import validate_lut

_lib = ctypes.CDLL(str(Path(__file__).parent / "local-results/native_fp4.so"))
_lib.exl3_pack.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
_lib.exl3_project.argtypes = (
    [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
)


_lib.exl3_had_pack.argtypes = (
    [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
)


_lib.exl3_arvq_pack.argtypes = _lib.exl3_had_pack.argtypes
_lib.exl3_arvq_project.argtypes = _lib.exl3_project.argtypes
_lib.exl3_arvq_project_four.argtypes = _lib.exl3_project.argtypes
_lib.exl3_fp4_gemv.argtypes = _lib.exl3_project.argtypes


def pack(x, input_scales=None, arvq=False):
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float16
    m, k = x.shape
    assert m > 0 and k % 64 == 0
    planes, group = (4, 16) if arvq else (2, 32)
    q = torch.empty((m, planes, k // 8), device=x.device, dtype=torch.int32)
    scales = torch.empty((m, planes, k // group), device=x.device, dtype=torch.uint8)
    with torch.accelerator.device_index(x.device.index):
        if arvq:
            assert k % 128 == 0
            if input_scales is not None:
                assert (
                    input_scales.shape == (k,) and input_scales.dtype == torch.float16
                )
                assert input_scales.device == x.device and input_scales.is_contiguous()
            code = _lib.exl3_arvq_pack(
                x.data_ptr(),
                input_scales.data_ptr() if input_scales is not None else None,
                q.data_ptr(),
                scales.data_ptr(),
                m,
                k,
                torch.cuda.current_stream().cuda_stream,
            )
        elif input_scales is None:
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


def project(
    x,
    packed,
    lut,
    splits=8,
    rows=None,
    scales=None,
    input_scales=None,
    arvq=False,
    gemv=False,
    weight_components=2,
):
    assert packed.is_contiguous() and packed.dtype == torch.int16
    validate_lut(lut, weight_components)
    if weight_components == 4 and not arvq:
        raise ValueError("Four weight components require four-plane ARVQ activations")
    assert x.device == packed.device == lut.device
    m, k = x.shape
    assert packed.shape[0] * 16 == k and packed.shape[2] in (24, 32, 40)
    n = packed.shape[1] * 16
    assert 1 <= splits <= k // 64
    if gemv:
        assert arvq and m >= 1 and packed.shape[2] == 32
    q, xs = pack(x, input_scales, arvq)
    scratch = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    with torch.accelerator.device_index(x.device.index):
        fn = (
            _lib.exl3_fp4_gemv
            if gemv
            else (_lib.exl3_arvq_project if arvq else _lib.exl3_project)
        )
        if weight_components == 4:
            fn = _lib.exl3_arvq_project_four
        code = fn(
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


_lib.exl3_fp4_fused.argtypes = (
    [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
)


def fused_project(x, packed, lut, input_scales, output_scales, splits=8, blocks=0):
    """Single cooperative launch; batch one, rate two, contiguous output."""
    assert x.is_cuda and x.dtype == torch.float16 and x.is_contiguous()
    m, k = x.shape
    assert m == 1 and k % 128 == 0
    assert packed.dtype == torch.int16 and packed.is_contiguous()
    assert packed.shape[0] * 16 == k and packed.shape[2] == 32
    n = packed.shape[1] * 16
    assert n % 128 == 0 and 1 <= splits <= k // 64
    assert lut.dtype == torch.uint8 and lut.numel() == 1024 and lut.is_contiguous()
    for scale, size in ((input_scales, k), (output_scales, n)):
        assert scale.shape == (size,) and scale.dtype == torch.float16
        assert scale.device == x.device and scale.is_contiguous()
    assert packed.device == lut.device == x.device
    q = torch.empty((4, k // 8), device=x.device, dtype=torch.int32)
    qs = torch.empty((4, k // 16), device=x.device, dtype=torch.uint8)
    partial = torch.empty((splits, n), device=x.device, dtype=torch.float32)
    out = torch.empty((1, n), device=x.device, dtype=torch.float32)
    with torch.accelerator.device_index(x.device.index):
        code = _lib.exl3_fp4_fused(
            x.data_ptr(),
            input_scales.data_ptr(),
            output_scales.data_ptr(),
            packed.data_ptr(),
            lut.data_ptr(),
            q.data_ptr(),
            qs.data_ptr(),
            partial.data_ptr(),
            out.data_ptr(),
            n,
            k,
            splits,
            blocks,
            torch.cuda.current_stream().cuda_stream,
        )
    if code:
        raise RuntimeError(f"exl3_fp4_fused CUDA error {code}")
    return out


_lib.exl3_fp4_cta_fused.argtypes = (
    [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
)


class FusedGEMV:
    """Explicit reusable workspace, restricted to one stream and batch one.

    A caller must not share this workspace between concurrent invocations.
    Counters return to zero after each successful invocation.
    """

    def __init__(self, packed, lut, input_scales, output_scales, splits):
        self.k, self.n = packed.shape[0] * 16, packed.shape[1] * 16
        assert packed.is_cuda and packed.dtype == torch.int16 and packed.is_contiguous()
        assert packed.shape[2] == 32 and self.k <= 6144
        assert self.k % 128 == self.n % 128 == 0
        assert 1 <= splits <= self.k // 64
        assert lut.dtype == torch.uint8 and lut.numel() == 1024 and lut.is_contiguous()
        assert lut.device == packed.device
        for scale, size in ((input_scales, self.k), (output_scales, self.n)):
            assert scale.dtype == torch.float16 and scale.shape == (size,)
            assert scale.is_contiguous() and scale.device == packed.device
        self.packed, self.lut = packed, lut
        self.su, self.sv = input_scales, output_scales
        self.splits = splits
        self.partial = torch.empty(
            (splits, self.n), device=packed.device, dtype=torch.float32
        )
        self.output = torch.empty(
            (1, self.n), device=packed.device, dtype=torch.float32
        )
        self.counters = torch.zeros(
            self.n // 128, device=packed.device, dtype=torch.int32
        )

    def __call__(self, x):
        assert x.dtype == torch.float16 and x.shape == (1, self.k) and x.is_contiguous()
        assert x.device == self.packed.device
        with torch.accelerator.device_index(x.device.index):
            stream = torch.cuda.current_stream().cuda_stream
            code = _lib.exl3_fp4_cta_fused(
                x.data_ptr(),
                self.su.data_ptr(),
                self.sv.data_ptr(),
                self.packed.data_ptr(),
                self.lut.data_ptr(),
                self.partial.data_ptr(),
                self.output.data_ptr(),
                self.counters.data_ptr(),
                self.n,
                self.k,
                self.splits,
                stream,
            )
        if code:
            raise RuntimeError(f"exl3_fp4_cta_fused CUDA error {code}")
        return self.output
