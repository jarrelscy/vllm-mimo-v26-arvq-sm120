# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU LUT checks and SM120 component/grouped parity checks."""

import argparse
import json

import torch
from weight_components import (
    SHIFTS,
    decode_lut,
    expert_luts,
    make_four_lut,
    validate_lut,
)


def table_checks(device):
    from trellis_gemm import make_pair_lut

    pair = make_pair_lut(device)
    four = make_four_lut(device)
    assert validate_lut(pair, 2) == 0
    assert validate_lut(four, 4) == 0
    for count, lut in ((2, pair), (4, four)):
        local = expert_luts(lut, count, 3)
        assert validate_lut(local, count, 3) == count // 2 * 1024
        torch.testing.assert_close(decode_lut(local, count)[0], decode_lut(lut, count))
    expected = torch.tensor([2.0**s for s in SHIFTS[4]])
    encoded = torch.tensor([0x38, 0x20, 0x08, 0x01], dtype=torch.uint8)
    assert torch.equal(encoded.view(torch.float8_e4m3fn).float(), expected)
    for lut, count in ((pair, 4), (four, 2), (four.float(), 4), (four, 3)):
        try:
            validate_lut(lut, count)
        except ValueError:
            continue
        raise AssertionError("Invalid LUT accepted")
    return pair, four


def states(packed):
    """Independent tensor implementation of the packed trellis bit layout."""
    k, n = packed.shape[0] * 16, packed.shape[1] * 16
    rate = packed.shape[-1] / 16
    ka, half = int(rate), int(rate % 1 != 0)
    kk = torch.arange(k, device=packed.device)[:, None]
    nn = torch.arange(n, device=packed.device)[None, :]
    r, c = kk % 16, nn % 16
    lane = c % 8 * 4 + r % 8 // 2
    pos = lane * 8 + r % 2 + 2 * (r // 8) + 4 * (c // 8)
    bits = 256 * ka + 128 * half
    words = bits // 32
    end = (pos + 1) * ka + (pos + 1) // 2 * half + bits
    tile = (kk // 16 * (n // 16) + nn // 16) * words
    raw = packed.view(torch.int32).flatten().long() & 0xFFFFFFFF
    a = raw[tile + ((end - 16) // 32) % words]
    b = raw[tile + ((end - 1) // 32) % words]
    state = torch.where(end % 32 == 0, b, ((a << 32) | b) >> (32 - end % 32)) & 65535
    product = state * 0x83DCD12D & 0xFFFFFFFF
    return (
        (product & 255)
        + ((product >> 8) & 255)
        + ((product >> 16) & 255)
        + (product >> 24)
    )


def operand_oracle(x, packed, lut, components):
    from native_fp4 import pack
    from weight_components import LEVELS

    q, scales = pack(x, arvq=True)
    levels = torch.tensor(LEVELS, device=x.device)
    codes = (q.long()[..., None] >> (torch.arange(8, device=x.device) * 4)) & 15
    codes = codes.flatten(-2)
    scale = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    activation = levels[codes] * scale
    activation /= torch.tensor([1, 16, 256, 4096], device=x.device)[None, :, None]
    indices = states(packed)
    table = lut.reshape(components // 2, 1024)
    result = torch.zeros(len(x), packed.shape[1] * 16, device=x.device)
    for component, shift in enumerate(SHIFTS[components]):
        digits = (table[component // 2].long() >> (4 * (component % 2))) & 15
        weight = levels[digits[indices]] * 2.0**shift
        for plane in range(4):
            result += activation[:, plane] @ weight
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    table_checks("cpu")
    if args.cpu_only:
        print("CPU codebook layout and E4M3 component-scale checks passed")
        return
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("Native MMA checks require SM120; use --cpu-only elsewhere")
    from grouped_fp4 import GroupedMoE, GroupedProjection
    from native_fp4 import project
    from sm120_grouped_confirm import reference

    torch.manual_seed(7129)
    torch.backends.cuda.matmul.allow_tf32 = False
    pair, four = table_checks("cuda")
    for count, lut in ((2, pair), (4, four)):
        for rate2 in (3, 4, 5):
            packed = torch.randint(
                -32768,
                32767,
                (16, 16, rate2 * 8),
                device="cuda",
                dtype=torch.int16,
            )
            for m in (1, 2, 3, 4):
                x = torch.randn(m, 256, device="cuda", dtype=torch.float16) * 0.1
                actual = project(
                    x, packed, lut, splits=1, arvq=True, weight_components=count
                )
                expected = operand_oracle(x, packed, lut, count)
                error = float((actual - expected).norm() / expected.norm())
                assert error < 2e-4, (count, rate2, m, error)

    # Distinct expert LUTs and permuted physical IDs exercise table addressing.
    e, k, n = 4, 256, 256
    packed = torch.randint(
        -32768,
        32767,
        (e, k // 16, n // 16, 32),
        device="cuda",
        dtype=torch.int16,
    )
    su = (torch.rand(e, k, device="cuda") + 0.5).half()
    sv = (torch.rand(e, n, device="cuda") + 0.5).half()
    x = torch.randn(6, k, device="cuda", dtype=torch.float16) * 0.01
    ids, counts = [3, 0, 2], [1, 3, 2]
    for count, shared in ((2, pair), (4, four)):
        local = expert_luts(shared, count, e).clone()
        for expert in range(e):
            local[expert] = local[expert].roll(expert * 7, dims=-1)
        grouped = GroupedProjection(
            packed, local, su, sv, ids, counts, 2, weight_components=count
        )
        actual = grouped(x).clone()
        outputs, offset = [], 0
        for expert, rows in zip(ids, counts):
            outputs.append(
                project(
                    x[offset : offset + rows],
                    packed[expert],
                    local[expert],
                    splits=2,
                    rows=torch.arange(n, device="cuda"),
                    scales=sv[expert],
                    input_scales=su[expert],
                    arvq=True,
                    weight_components=count,
                )
            )
            offset += rows
        expected = torch.cat(outputs)
        error = float((actual - expected).norm() / expected.norm())
        assert error < 2e-4, (count, error)
        graph = torch.cuda.CUDAGraph()
        for _ in range(3):
            grouped(x)
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            grouped(x)
        graph.replay()
        assert torch.equal(grouped.output, actual)

    # Existing shared-two-component MoE API remains valid.
    parts = [(packed, su, sv)] * 3
    moe = GroupedMoE(*parts, pair, 2, topk=2, splits_gu=2, splits_down=2)
    x = x[:2]
    chosen = torch.tensor([[3, 0], [2, 3]], device="cuda")
    routing = torch.full((2, 2), 0.5, device="cuda", dtype=torch.float16)
    expected = reference(parts, pair, x, chosen.tolist(), routing, (2, 2))
    actual = moe(x, chosen, routing)
    assert float((actual - expected).norm() / expected.norm()) < 2e-4
    # Separate learned tables for gate/up/down and each physical expert.
    for count, shared in ((2, pair), (4, four)):
        tables = []
        for projection in range(3):
            local = expert_luts(shared, count, e).clone()
            for expert in range(e):
                local[expert] = local[expert].roll(
                    5 * expert + 11 * projection, dims=-1
                )
            tables.append(local)
        moe = GroupedMoE(
            *parts,
            tuple(tables),
            2,
            topk=2,
            splits_gu=2,
            splits_down=2,
            weight_components=count,
        )
        expected = torch.zeros_like(x, dtype=torch.float32)
        for token, experts in enumerate(chosen.tolist()):
            for slot, expert in enumerate(experts):
                gu = [
                    project(
                        x[token : token + 1],
                        p[expert],
                        tables[j][expert],
                        splits=2,
                        rows=torch.arange(n, device=x.device),
                        scales=v[expert],
                        input_scales=u[expert],
                        arvq=True,
                        weight_components=count,
                    )
                    for j, (p, u, v) in enumerate(parts[:2])
                ]
                hidden = (torch.nn.functional.silu(gu[0]) * gu[1]).half()
                p, u, v = parts[2]
                output = project(
                    hidden,
                    p[expert],
                    tables[2][expert],
                    splits=2,
                    rows=torch.arange(k, device=x.device),
                    scales=v[expert],
                    input_scales=u[expert],
                    arvq=True,
                    weight_components=count,
                )
                expected[token : token + 1] += output * routing[token, slot].float()
        actual = moe(x, chosen, routing).clone()
        assert float((actual - expected).norm() / expected.norm()) < 2e-4
        for _ in range(3):
            moe(x, chosen, routing)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe(x, chosen, routing)
        graph.replay()
        assert torch.equal(moe.output, actual)
    print(json.dumps({"status": "passed", "weight_components": [2, 4]}))


if __name__ == "__main__":
    main()
