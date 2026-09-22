# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse identical input-plane quantization across routed experts."""

from arvq88.activation import activation_ste, swiglu_ste


def expert_from_packed_input(packed_input, w13, w2):
    return activation_ste(swiglu_ste(packed_input @ w13.T)) @ w2.T
