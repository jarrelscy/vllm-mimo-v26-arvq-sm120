# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped rate-2 native FP4 projection with explicit reusable workspace."""

import ctypes

import torch
from native_fp4 import _lib
from weight_components import expert_luts, validate_lut

_lib.exl3_grouped_pack.argtypes = (
    [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
)
_lib.exl3_grouped_project.argtypes = (
    [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
)
_lib.exl3_grouped_project_components.argtypes = (
    [ctypes.c_void_p] * 8 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
)
_lib.exl3_grouped_reduce.argtypes = (
    [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
)


class GroupedProjection:
    """Caller supplies expert-grouped rows; workspace may not run concurrently."""

    def __init__(
        self, packed, lut, su, sv, expert_ids, counts, splits, weight_components=2
    ):
        assert packed.is_cuda and packed.dtype == torch.int16 and packed.is_contiguous()
        assert packed.ndim == 4 and packed.shape[-1] == 32
        self.k, self.n = packed.shape[1] * 16, packed.shape[2] * 16
        assert self.k % 128 == self.n % 128 == 0 and 1 <= splits <= self.k // 64
        assert len(expert_ids) == len(counts) > 0 and all(c > 0 for c in counts)
        assert all(0 <= e < packed.shape[0] for e in expert_ids)
        assert su.shape == (packed.shape[0], self.k) and sv.shape == (
            packed.shape[0],
            self.n,
        )
        for scale in [su, sv]:
            assert (
                scale.dtype == torch.float16
                and scale.is_contiguous()
                and scale.device == packed.device
            )
        assert (
            lut.dtype == torch.uint8
            and lut.is_contiguous()
            and lut.device == packed.device
        )
        self.weight_components = weight_components
        self.lut_stride = validate_lut(lut, weight_components, packed.shape[0])
        self.packed, self.lut, self.su, self.sv = packed, lut, su, sv
        self.counts = counts
        self.e, self.r, self.max_m, self.s = (
            len(counts),
            sum(counts),
            max(counts),
            splits,
        )
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        self.ids = torch.tensor(expert_ids, device=packed.device, dtype=torch.int32)
        self.offsets = torch.tensor(offsets, device=packed.device, dtype=torch.int32)
        self.groups = torch.tensor(
            [i for i, c in enumerate(counts) for _ in range(c)],
            device=packed.device,
            dtype=torch.int32,
        )
        self.q = torch.empty(
            (self.r, 4, self.k // 8), device=packed.device, dtype=torch.int32
        )
        self.qs = torch.empty(
            (self.r, 4, self.k // 16), device=packed.device, dtype=torch.uint8
        )
        self.partial = torch.empty(
            (self.r * self.s, self.n), device=packed.device, dtype=torch.float32
        )
        self.output = torch.empty(
            (self.r, self.n), device=packed.device, dtype=torch.float32
        )

    def project_packed(self, stream):
        args = (
            self.packed.data_ptr(),
            self.lut.data_ptr(),
            self.q.data_ptr(),
            self.qs.data_ptr(),
            self.ids.data_ptr(),
            self.offsets.data_ptr(),
            self.groups.data_ptr(),
            self.partial.data_ptr(),
            self.r,
            self.n,
            self.k,
            self.s,
        )
        if self.weight_components == 2 and self.lut_stride == 0:
            return _lib.exl3_grouped_project(*args, stream)
        return _lib.exl3_grouped_project_components(
            *args, self.weight_components, self.lut_stride, stream
        )

    def __call__(self, x):
        assert (
            x.shape == (self.r, self.k)
            and x.dtype == torch.float16
            and x.is_contiguous()
        )
        assert x.device == self.packed.device
        with torch.accelerator.device_index(x.device.index):
            stream = torch.cuda.current_stream().cuda_stream
            code = _lib.exl3_grouped_pack(
                x.data_ptr(),
                self.su.data_ptr(),
                self.ids.data_ptr(),
                self.groups.data_ptr(),
                self.q.data_ptr(),
                self.qs.data_ptr(),
                self.r,
                self.k,
                stream,
            )
            if code:
                raise RuntimeError(f"grouped pack CUDA error {code}")
            code = self.project_packed(stream)
            if code:
                raise RuntimeError(f"grouped project CUDA error {code}")
            code = _lib.exl3_grouped_reduce(
                self.partial.data_ptr(),
                self.sv.data_ptr(),
                self.ids.data_ptr(),
                self.offsets.data_ptr(),
                self.groups.data_ptr(),
                self.output.data_ptr(),
                self.r,
                self.n,
                self.s,
                stream,
            )
            if code:
                raise RuntimeError(f"grouped reduce CUDA error {code}")
        return self.output


_lib.exl3_route_metadata.argtypes = (
    [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
)
_lib.exl3_routed_pack.argtypes = (
    [ctypes.c_void_p] * 7 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
)
_lib.exl3_grouped_swiglu.argtypes = (
    [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
)
_lib.exl3_grouped_sum.argtypes = (
    [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
)


class GroupedMoE:
    """GPU routing metadata and grouped FP4 MLP with reusable output/workspace.

    Calls must not overlap. Router IDs must be in [0, E), with distinct IDs
    within each token's top-k. Valid routing is a caller precondition, as in
    the official EXL3 adapter. The returned tensor is overwritten next call.
    """

    def __init__(
        self,
        gate,
        up,
        down,
        lut,
        m,
        topk=8,
        splits_gu=None,
        splits_down=None,
        weight_components=2,
    ):
        self.e = gate[0].shape[0]
        self.h = gate[0].shape[1] * 16
        self.i = gate[0].shape[2] * 16
        self.m, self.topk, self.r = m, topk, m * topk
        assert 1 <= m <= 4 and 1 <= topk <= self.e and self.r <= 32
        assert len(gate) == len(up) == len(down) == 3
        assert all(a.shape == b.shape for a, b in zip(gate, up))
        assert down[0].shape == (self.e, self.i // 16, self.h // 16, 32)
        assert all(
            t.device == gate[0].device for part in [gate, up, down] for t in part
        )
        # Fixed TP4 policy, selected on an independent tuning seed.
        policy = {1: (8, 4), 2: (4, 1), 3: (8, 1), 4: (8, 1)}[m]
        splits_gu = min(policy[0], self.h // 64) if splits_gu is None else splits_gu
        splits_down = (
            min(policy[1], self.i // 64) if splits_down is None else splits_down
        )
        self.gu_data = tuple(
            torch.cat([a, b], 0).contiguous() for a, b in zip(gate, up)
        )
        self.down_data = down
        if isinstance(lut, (tuple, list)):
            if len(lut) != 3:
                raise ValueError("Supply gate, up and down codebooks")
            gate_lut, up_lut, down_lut = (
                expert_luts(value, weight_components, self.e) for value in lut
            )
            gu_lut = torch.cat((gate_lut, up_lut), 0).contiguous()
        else:
            gu_lut = down_lut = lut
            if validate_lut(lut, weight_components, self.e):
                local = expert_luts(lut, weight_components, self.e)
                gu_lut = torch.cat((local, local), 0).contiguous()
                down_lut = local
        counts = [1] * self.r
        self.gu = GroupedProjection(
            self.gu_data[0],
            gu_lut,
            self.gu_data[1],
            self.gu_data[2],
            [0] * (2 * self.r),
            counts + counts,
            splits_gu,
            weight_components=weight_components,
        )
        self.down = GroupedProjection(
            down[0],
            down_lut,
            down[1],
            down[2],
            [0] * self.r,
            counts,
            splits_down,
            weight_components=weight_components,
        )
        self.gu.max_m = self.down.max_m = m
        self.ids, self.offsets, self.groups = (
            self.gu.ids,
            self.gu.offsets,
            self.gu.groups,
        )
        self.down.ids = self.ids[: self.r]
        self.down.offsets = self.offsets[: self.r + 1]
        self.down.groups = self.groups[: self.r]
        dev = gate[0].device
        self.gather = torch.empty(2 * self.r, device=dev, dtype=torch.int32)
        self.inverse = torch.empty(self.r, device=dev, dtype=torch.int32)
        self.activation = torch.empty((self.r, self.i), device=dev, dtype=torch.float16)
        self.output = torch.empty((m, self.h), device=dev, dtype=torch.float32)

    def __call__(self, x, selected, weights):
        assert (
            x.shape == (self.m, self.h)
            and x.dtype == torch.float16
            and x.is_contiguous()
        )
        assert selected.shape == weights.shape == (self.m, self.topk)
        assert selected.dtype == torch.int64 and weights.dtype == torch.float16
        assert selected.is_contiguous() and weights.is_contiguous()
        assert x.device == selected.device == weights.device == self.output.device
        with torch.accelerator.device_index(x.device.index):
            stream = torch.cuda.current_stream().cuda_stream
            code = _lib.exl3_route_metadata(
                selected.data_ptr(),
                self.ids.data_ptr(),
                self.offsets.data_ptr(),
                self.groups.data_ptr(),
                self.gather.data_ptr(),
                self.inverse.data_ptr(),
                self.m,
                self.topk,
                self.e,
                stream,
            )
            if code:
                raise RuntimeError(f"route metadata error {code}")
            for stage, inputs in [(self.gu, x), (self.down, self.activation)]:
                code = _lib.exl3_routed_pack(
                    inputs.data_ptr(),
                    stage.su.data_ptr(),
                    stage.ids.data_ptr(),
                    stage.groups.data_ptr(),
                    self.gather.data_ptr() if stage is self.gu else None,
                    stage.q.data_ptr(),
                    stage.qs.data_ptr(),
                    stage.r,
                    stage.k,
                    stream,
                )
                if code:
                    raise RuntimeError(f"routed pack error {code}")
                code = stage.project_packed(stream)
                if code:
                    raise RuntimeError(f"grouped project error {code}")
                code = _lib.exl3_grouped_reduce(
                    stage.partial.data_ptr(),
                    stage.sv.data_ptr(),
                    stage.ids.data_ptr(),
                    stage.offsets.data_ptr(),
                    stage.groups.data_ptr(),
                    stage.output.data_ptr(),
                    stage.r,
                    stage.n,
                    stage.s,
                    stream,
                )
                if code:
                    raise RuntimeError(f"grouped reduce error {code}")
                if stage is self.gu:
                    code = _lib.exl3_grouped_swiglu(
                        stage.output.data_ptr(),
                        self.activation.data_ptr(),
                        self.r,
                        self.i,
                        stream,
                    )
                    if code:
                        raise RuntimeError(f"grouped activation error {code}")
            code = _lib.exl3_grouped_sum(
                self.down.output.data_ptr(),
                weights.data_ptr(),
                self.inverse.data_ptr(),
                self.output.data_ptr(),
                self.m,
                self.topk,
                self.h,
                stream,
            )
            if code:
                raise RuntimeError(f"grouped sum error {code}")
        return self.output
