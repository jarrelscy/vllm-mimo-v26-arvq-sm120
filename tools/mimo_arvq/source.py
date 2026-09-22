# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read individual released MiMo experts without loading the whole model."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def decode_mxfp4(packed, scales):
    """Decode row-major E2M1 pairs with one E8M0 scale per 32 weights."""
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("Expected packed U8 weights and E8M0 scale bytes")
    if packed.ndim != 2 or scales.shape != (packed.shape[0], packed.shape[1] // 16):
        raise ValueError("Unexpected MXFP4 weight/scale shape")
    if packed.shape[1] % 16:
        raise ValueError("Incomplete MXFP4 block")
    if torch.any(scales == 255):
        raise ValueError("Source contains reserved NaN E8M0 scale")
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    factor = torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32), scales.to(torch.int32) - 127
    )
    return lut[codes.long()] * factor.repeat_interleave(32, dim=-1)


class Source:
    """Keep the original index and tensor names as the source of truth."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.index = json.loads(
            (self.directory / "model.safetensors.index.json").read_text()
        )["weight_map"]

    def tensor(self, name, device="cpu"):
        with safe_open(
            self.directory / self.index[name], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name).to(device)

    def expert(self, layer, expert, projection, device="cpu"):
        if not 0 <= layer < self.config["num_hidden_layers"]:
            raise ValueError("Invalid layer")
        if not self.config["moe_layer_freq"][layer]:
            raise ValueError("Requested a dense layer")
        if not 0 <= expert < self.config["n_routed_experts"]:
            raise ValueError("Invalid expert")
        if projection not in ("gate_proj", "up_proj", "down_proj"):
            raise ValueError("Invalid projection")
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        weight = self.tensor(prefix + ".weight", device)
        scale = self.tensor(prefix + ".weight_scale", device)
        return decode_mxfp4(weight, scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = Source(args.source)
    report = {"layer": args.layer, "expert": args.expert, "projections": {}}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weight = source.expert(args.layer, args.expert, projection, args.device)
        if not torch.isfinite(weight).all():
            raise ValueError("Non-finite source expert")
        expected = [
            source.config["moe_intermediate_size"],
            source.config["hidden_size"],
        ]
        if projection == "down_proj":
            expected.reverse()
        if list(weight.shape) != expected:
            raise ValueError("Decoded source expert shape mismatch")
        report["projections"][projection] = {
            "shape": list(weight.shape),
            "rms": weight.square().mean().sqrt().item(),
            "abs_max": weight.abs().max().item(),
            "finite": True,
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
