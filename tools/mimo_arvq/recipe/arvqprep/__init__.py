"""ARVQ (additive FP4 residual-VQ) cold-tier encoder for the GLM-5.3 hybrid.

Mirrors btxprep/ but targets the SM120 NVFP4-ARVQ hybrid loader
(vllm-glm52-sm120 @ arvq-hybrid-sm120). Format `rvq256_128x8`:

  W[n, 8g:8g+8] = global * s[n, block(g)] * (c0[a_{n,g}] + c1[b_{n,g}])

  - c0: 256 x 8 FP4-constrained vectors, c1: 128 x 8 (shared per layer/proj)
  - a: uint8 index (8 bit), b: uint8 index (<128, 7 bit); 15-bit pair / 8 weights
  - s: per (output row, 128-input-col block) E4M3 scale
  - global: fp32 scalar per (layer, projection)

No incoherence rotation (the kernel feeds raw activations); Hessians are built
in the raw weight basis. Codebooks are shared across all cold experts of a
(layer, projection), so the fit is a per-layer joint optimization.
"""
from __future__ import annotations

# FP4 / e2m1 decode levels (order MUST match the branch's
# examples/arvq/convert/fit_codebooks.py::LEVELS and format_utils.pack_cb).
LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
