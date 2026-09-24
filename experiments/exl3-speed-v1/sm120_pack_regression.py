# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from pathlib import Path

import torch
from native_fp4 import _lib, pack

old = ctypes.CDLL(str(Path(__file__).parent / "local-results/baseline_a187.so"))
old.exl3_arvq_pack.argtypes = _lib.exl3_arvq_pack.argtypes
v = torch.arange(65536, device="cuda", dtype=torch.int32).short().view(torch.float16)
v = v[torch.isfinite(v)]
for shuffle in [False, True]:
    if shuffle:
        v = v[torch.randperm(v.numel(), device="cuda")]
    x = v.reshape(-1, 128).contiguous()
    a, s = pack(x, None, True)
    b = torch.empty_like(a)
    t = torch.empty_like(s)
    code = old.exl3_arvq_pack(
        x.data_ptr(),
        None,
        b.data_ptr(),
        t.data_ptr(),
        x.shape[0],
        128,
        torch.cuda.current_stream().cuda_stream,
    )
    assert code == 0
    assert torch.equal(a, b) and torch.equal(s, t)
    print("all finite FP16 pack exact", shuffle, x.numel(), flush=True)
for k in [128, 256, 2048, 6144]:
    x = torch.randn(7, k, device="cuda", dtype=torch.float16) * 10
    su = torch.randn(k, device="cuda", dtype=torch.float16)
    a, s = pack(x, su, True)
    b = torch.empty_like(a)
    t = torch.empty_like(s)
    code = old.exl3_arvq_pack(
        x.data_ptr(),
        su.data_ptr(),
        b.data_ptr(),
        t.data_ptr(),
        7,
        k,
        torch.cuda.current_stream().cuda_stream,
    )
    assert code == 0
    assert torch.equal(a, b) and torch.equal(s, t)
    print("fused pack exact", k, flush=True)
