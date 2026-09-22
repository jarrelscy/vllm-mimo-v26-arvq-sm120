# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify capture attention against explicit softmax on real layer inputs."""

import argparse
import json
from pathlib import Path

import torch
from reference import Layer
from source import Source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = Source(args.work / "source")
    if source.config["n_group"] != 1 or source.config["topk_group"] != 1:
        raise ValueError("Capture routing requires the released single-group model")
    embedding = source.tensor("model.embed_tokens.weight", "cuda:0").to(torch.bfloat16)
    tokens = torch.arange(32, device="cuda:0")[None, :] + 100
    state = torch.nn.functional.embedding(tokens, embedding)
    del embedding
    results = []
    with torch.no_grad():
        for number in (0, 1, 7):
            layer = Layer(source, number, "cuda:0")
            fast, x_fast = layer.front(state)
            eager, x_eager = layer.front(state, eager=True)
            error = float((fast.float() - eager.float()).norm() / eager.float().norm())
            input_error = float(
                (x_fast.float() - x_eager.float()).norm() / x_eager.float().norm()
            )
            if not torch.isfinite(fast).all() or max(error, input_error) > 0.005:
                raise RuntimeError(
                    f"Capture attention parity failed: {number}: {error}, {input_error}"
                )
            if number:
                bias = source.tensor(
                    layer.prefix + "mlp.gate.e_score_correction_bias", "cuda:0"
                )
                torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)
                assert layer.bias.dtype == torch.float32
                inputs = x_fast.flatten(0, 1)
                scores = torch.nn.functional.linear(
                    inputs.float(), layer.gate.float()
                ).sigmoid()
                expected_ids = (scores + bias).topk(8, dim=-1, sorted=False).indices
                expected_gates = scores.gather(-1, expected_ids)
                expected_gates /= expected_gates.sum(-1, keepdim=True) + 1e-20
                actual_ids, actual_gates = layer.route(inputs)
                torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(actual_gates, expected_gates, rtol=0, atol=0)
            results.append(
                {
                    "layer": number,
                    "residual_relative_error": error,
                    "normalized_input_relative_error": input_error,
                }
            )
            del layer
    result = {
        "passed": True,
        "router_fp32_bias_qualified": True,
        "results": results,
        "scope": "SDPA capture versus explicit softmax; not full-model or SM120 parity",
    }
    (args.work / "capture_attention_qualified.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
