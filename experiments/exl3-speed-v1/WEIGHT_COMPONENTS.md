# Locally fitted two- and four-component FP4 weights

The native experimental kernel accepts shared or per-expert codebooks, with
separate tables for gate, up and down. Set `weight_components=4` to use four
FP4 weight components. The default remains two components. Both paths use
native FP4 MMA; neither reconstructs an FP16 weight matrix.

## Representation

A trellis state hashes to one of 1,024 table slots. Each slot holds two or four
FP4 E2M1 values. Weight reconstruction is:

- Two components: `a + b/16`.
- Four components: `a + b/8 + c/64 + d/512`.

A shared table costs 1 KiB or 2 KiB respectively. Local tables incur that cost
per expert and projection. Components do not add trellis index bits, but table
and scale metadata must be counted in total bits per weight.

LUTs are contiguous CUDA `uint8` tensors. Each byte packs component `2*p` in
its low nibble and component `2*p+1` in its high nibble. Shapes are:

| Scope | Two components | Four components |
| --- | --- | --- |
| Shared | `[1024]` or `[1,1024]` | `[2,1024]` |
| Per expert | `[E,1,1024]` or `[E,1024]` | `[E,2,1024]` |

Expert indices are physical expert IDs, independent of routing order. For a
fitter's unpacked `[1024,4]` nibble table, pack it with:

```python
lut = (digits[:, ::2] | (digits[:, 1::2] << 4)).T.contiguous()
```

Fit against the selected FP4 reconstruction directly. `make_four_lut()` is
only a reference approximation of the original table, not a learned codebook.

## API

```python
from grouped_fp4 import GroupedMoE

moe = GroupedMoE(
    gate, up, down,
    (gate_luts, up_luts, down_luts),  # each [E,2,1024], uint8 CUDA
    m=batch_size,
    topk=8,
    weight_components=4,
)
y = moe(x, expert_ids, routing_weights)
```

Each projection tuple is `(packed_weights, input_scales, output_scales)`, as in
the existing API. A single LUT tensor is also accepted and reused across all
three projections. Tables can be locally fitted for two components as well.

`GroupedProjection(..., weight_components=4)` accepts a shared or per-expert
LUT. `native_fp4.project(..., arvq=True, weight_components=4)` accepts one
shared table for a standalone projection. Four-component execution currently
requires the existing four-plane activation representation.

Grouped execution supports rate-2 packed weights. Standalone execution
supports rates 1.5, 2 and 2.5. Mixed-rate pilot artifacts still require an
additional grouped segment adapter; this change is not a full checkpoint loader.

## Evidence and limitations

Layer 1, expert 42, initial direct-FP4 fit, A100 operand emulation:

| Codebook | Components | Validation error | Audit error |
| --- | --- | --- | --- |
| Shared | 2 | 15.42% | 13.20% |
| Shared | 4 | 16.48% | 16.03% |
| Locally fitted | 2 | 15.17% | 11.01% |
| Locally fitted | 4 | 14.13% | 10.90% |

These results precede runtime-aware PV and native SM120 validation. Expert 0
underflowed its down-projection activations in every variant (100% error), so
there is only one usable expert comparison. Four local components are a
candidate, not a demonstrated model-wide winner. The activation scale issue
is not fixed by this kernel change.

Four-component execution currently makes two component-pair passes, repeating
trellis decoding and accumulating into the same FP32 partial buffer. It is a
correctness implementation pending native testing and speed tuning. No new
SM120 speed result is claimed. The existing shared two-component path remains
the default and uses its original dispatch.

## Validation

Cross-compilation for SM120a and CPU LUT/scale checks passed on the A100 host.
Native MMA parity, graph replay and sanitizer checks require an SM120 device:

```bash
bash experiments/exl3-speed-v1/build_native_fp4.sh
.venv/bin/python experiments/exl3-speed-v1/sm120_weight_components_check.py --cpu-only
.venv/bin/python experiments/exl3-speed-v1/sm120_weight_components_check.py
compute-sanitizer --tool memcheck .venv/bin/python experiments/exl3-speed-v1/sm120_weight_components_check.py
```

The native checker covers standalone component/rate combinations, distinct
per-expert tables with permuted routing, separate gate/up/down tables, grouped
MoE, CUDA graph replay, and the legacy shared two-component API. Its standalone
oracle independently decodes packed trellis states and multiplies FP4 operands.
