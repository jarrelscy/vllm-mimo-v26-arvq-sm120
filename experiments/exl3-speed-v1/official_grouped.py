# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark adapter for official EXL3's two-launch small-batch MoE path."""

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3


class OfficialGrouped:
    def __init__(self, gate, up, down, topk=8):
        self.data = [gate, up, down]
        E = gate[0].shape[0]
        H = gate[0].shape[1] * 16
        intermediate = gate[0].shape[2] * 16
        device = gate[0].device
        R = 8 * topk
        self.layers = []
        ptrs = []
        for p, su, sv in self.data:
            k, n = p.shape[1] * 16, p.shape[2] * 16
            self.layers.append(
                [
                    LinearEXL3(
                        None,
                        k,
                        n,
                        trellis=p[e],
                        suh=su[e],
                        svh=sv[e],
                        mul1=torch.tensor(True, device=device),
                    )
                    for e in range(E)
                ]
            )
            ptrs.append(
                [
                    torch.tensor(
                        [a[e].data_ptr() for e in range(E)],
                        device=device,
                        dtype=torch.int64,
                    )
                    for a in [p, su, sv]
                ]
            )
        hidden = torch.empty((R, H), device=device, dtype=torch.float16)
        interm = torch.empty((2 * R, intermediate), device=device, dtype=torch.float32)
        activation = torch.empty((R, intermediate), device=device, dtype=torch.float16)
        output = torch.empty((R, H), device=device, dtype=torch.float32)
        self.out = torch.empty((8, H), device=device, dtype=torch.float32)
        counters = torch.zeros(
            R * (intermediate // 128) + 8 * (H // 128) + 2 * R + 3,
            device=device,
            dtype=torch.int32,
        )
        had_u = torch.empty_like(hidden)
        dq = torch.empty((H, intermediate), device=device, dtype=torch.float16)
        kw = dict(
            yh2=hidden,
            yh=hidden.view(R, 1, H),
            interm_gu=interm,
            interm_g=interm[:R].view(R, 1, intermediate),
            interm_u=interm[R:].view(R, 1, intermediate),
            interm_a=activation.view(R, 1, intermediate),
            interm_a2=activation,
            out_d=output.view(R, 1, H),
            out_d2=output,
            out_d_sh=None,
            coop_ctr=counters,
            had_u=had_u,
            dq_temp_up=dq,
            dq_temp_down=dq.view(intermediate, H),
            min_expert=-1,
            max_expert=-1,
            act_silu=True,
            act_gelu=False,
            act_silu_oai=False,
            shared_experts=None,
            shared_gate=None,
            act_limit=0.0,
            gates=[x.bc for x in self.layers[0]],
            ups=[x.bc for x in self.layers[1]],
            downs=[x.bc for x in self.layers[2]],
            gu_trellis_ptr=torch.stack([ptrs[0][0], ptrs[1][0]], 1).contiguous(),
            gu_suh_ptr=torch.stack([ptrs[0][1], ptrs[1][1]], 1).contiguous(),
            gu_svh_ptr=torch.stack([ptrs[0][2], ptrs[1][2]], 1).contiguous(),
            out_bszn=self.out,
        )
        for label, ps in zip(["gate", "up", "down"], ptrs):
            kw.update(
                {
                    f"{label}_ptrs_trellis": ps[0],
                    f"{label}_ptrs_suh": ps[1],
                    f"{label}_ptrs_svh": ps[2],
                    f"{label}_K": 2.0,
                    f"{label}_mcg": False,
                    f"{label}_mul1": True,
                }
            )
        self.bc = ext.BC_BlockSparseMLP(**kw)

    def __call__(self, x, selected, weights):
        self.bc.run_bszN(x, selected, weights)
        return self.out[: x.shape[0]]
