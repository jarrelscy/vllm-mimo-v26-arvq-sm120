# Native SM120 FP4 experiment

Measured on RTX PRO 6000 Blackwell Max-Q, 24 September 2026. This is a
synthetic single-projection experiment, not a fitted-model quality test,
full MoE benchmark, or vLLM integration. The serving model remained resident
but idle; tests used a separate process on GPU 1. Warm CUDA graph timings
include input transform/scales, activation packing, decode/MMA, reduction,
and output transform/scales unless explicitly marked as individual operations.

## What executes

The original Triton `gemm_fp4` at BM16/BN64 emits BF16 tensor instructions on
this SM120 configuration. PTX contains `mma.sync...bf16.bf16`; SASS contains
`HMMA.16816.F32.BF16`. Its FP4 operand names do not imply native FP4 execution.

`native_fp4.cu` instead emits
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0`.
SASS confirms `OMMA.SF.16864.F32.E2M1.E2M1.E8`. Build with the explicit
architecture-specific `-gencode` below; CUDA 12.9's `-arch=sm_120a` alone
selected an incompatible virtual target in this environment.

The native path retains the original packed trellis layout and supports
1.5/2/2.5-bit rates. It uses the 1 KiB two-FP4-component lookup, two independently
scaled activation planes with groups of 32, and four native MMA terms.
No expanded weight matrix is stored. FP32 split-K partials are temporary.
The optional input Hadamard/scale/packing fusion preserves the official
FP16 pre-scale rounding before the FP32 transform and FP16 output boundary.

This is **not lossless relative to FP16 EXL3**: the weight lookup approximates
the original codebook and activation packing adds quantization. Tests instead
establish parity with the declared FP4 operands. Activation midpoint ties go
toward zero; the earlier signed sorted-level emulation uses a different rule
for negative ties. It also rounds the reconstructed activation sum to FP16;
native MMA accumulates the separate planes without that extra boundary.
Real-checkpoint quality therefore needs a new audit.

## Results and limits

The JSON files in `results/sm120-20260924/` contain all tested batches and
split-K choices, including rejected experiments. `official` compares direct
and reconstruct-each-call; use the faster official path at each shape.
These are independent projections, not gate/up/down summed model throughput.

Batch-1 complete projection latency, microseconds:

| K → N | Best official | Custom FP16 | Triton FP4 fallback | Native FP4 | Native splits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6144 → 512 | 8.21 | 16.41 | 24.58 | 12.29 | 96 |
| 512 → 6144 | 6.17 | 12.29 | 16.04 | 11.07 | 8 |
| 6144 → 2048 | 10.25 | 22.63 | 34.86 | 22.55 | 96 |
| 2048 → 6144 | 8.21 | 20.00 | 28.68 | 22.53 | 32 |

Hadamard alone is approximately 1.8–2.1 microseconds at small batches.
At batch 1, combining output reduction/Hadamard/scatter is about 2.1 us,
versus about 4.1 us for separate reduction and Hadamard without scatter.
The existing fused output kernel becomes slower at some larger batches.

The native kernel remains slower than the best official EXL3 path. It is a
working hardware baseline, not a speedup claim. More split-K parallelism
helps narrow-output decode. Keeping the small lookup in read-only device
memory avoids per-CTA table initialization and improves several shapes.
Cooperative warp-shuffle loading and a 64 KiB direct-state table were tested
and rejected: their additional shuffle/cache costs erased the expected gains.
Input Hadamard fusion saves a launch but is a small improvement on its own.

Next targets are fragment-aware extraction of overlapping trellis states,
reusing decoded fragments across token tiles, and pipelining decode with MMA.
The diagnosis of decode/packing overhead is inferred from these experiments;
no hardware-counter bottleneck profile has been completed.

## Validation

- Published official EXL3 synthetic smoke test passes all three rates and
  batches 1/4/16/64 on SM120.
- Independent bit-by-bit integer oracle passes exact and FP4-pair decoding.
- Native identity reconstruction exactly matches the FP4-pair codewords.
- Native random-input reference covers all three rates, batches
  1/4/7/8/9/16/64, and activation magnitudes spanning five orders.
  Maximum relative L2 error is approximately 5.4e-8; maximum absolute error
  is 1.22e-4 for these scaled inputs.
- Additional checks cover partial output tiles and uneven split-K ranges.
- Fused input transform/packing matches official Hadamard followed by native
  packing bit-for-bit on random nonuniform scales at batches 1/9/64.
- Compute Sanitizer memcheck reports zero errors (expandable allocator disabled
  because this container denies the virtual-memory allocation API under memcheck).
- CUDA graphs are exercised by all timing drivers. Real checkpoint quality,
  routed expert batching, TP communication, and end-to-end serving are untested.

## Reproduce

Use a virtual environment with torch 2.11.0+cu130, Triton, and official
EXL3 1.5.1+cu128.torch2.11.0. This test used CUDA 12.9 nvcc.

```bash
NVCC=/usr/local/cuda/bin/nvcc bash build_native_fp4.sh
.venv/bin/python smoke_test.py
.venv/bin/python sm120_verify.py
.venv/bin/python sm120_microbench.py
.venv/bin/python sm120_microbench.py --fp4
.venv/bin/python sm120_official_bench.py
.venv/bin/python sm120_native_test.py --fused
PYTORCH_ALLOC_CONF=expandable_segments:False \
  compute-sanitizer --tool memcheck --error-exitcode 99 \
  .venv/bin/python sm120_native_test.py --parity-only
cuobjdump --dump-sass local-results/native_fp4.so
```

Scripts write local results to the git-ignored `local-results/` directory.
The published JSON is a snapshot, not regenerated implicitly by the tests.

## Four activation planes and specialized batch-one kernel

The ARVQ-style variant places four residual activation planes in MMA columns,
using group-16 E4M3 power-of-two scales and two weight-plane MMA instructions.
Two tokens fill all eight columns. SASS confirms native
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. The generic version did not improve
batch-one speed. Against the independently reconstructed FP4-plane reference,
maximum relative L2 was 6.48e-8 and maximum absolute error was 2.44e-4.

`project(..., arvq=True, gemv=True)` selects a new rate-2, batch-one
specialization. It extracts four related trellis states from one pair of loaded
words and assembles both weight-plane MMA fragments from them. Other rates or
batch sizes are rejected by this explicit entry point; the generic path remains
available. This is not a fully fused projection: activation packing and the
output reduction/transform remain separate launches.

Paired complete-projection timings on the same SM120, microseconds:

| K → N | Generic four-plane | Specialized four-plane | Official EXL3 |
| --- | ---: | ---: | ---: |
| 6144 → 512 | 14.12 | 10.26 | 8.21 |
| 512 → 6144 | 12.29 | 10.25 | 6.17 |
| 6144 → 2048 | 24.58 | 16.39 | 10.24 |
| 2048 → 6144 | 24.58 | 16.39 | 8.19 |

Rows use the fastest tested specialized split count (96, 8, 96, 16).
The specialization matches the generic FP4 result bit-for-bit across all
six tested shapes and valid split counts, including partial output tiles and
uneven splits. Compute Sanitizer reports zero errors. This verifies the
implementation against the same quantized operands, not losslessness against
original EXL3 weights or activations. See `sm120_gemv_test.py` and the corresponding
JSON snapshot. The kernel is experimental and is not wired into serving.

Correction to the earlier description of the official baseline: profiling
`LinearEXL3.forward` on this RTX PRO 6000 (compute capability 12.0), batch one,
rate two, 6144→2048, captured
`exl3_gemv_int8_sq_kernel<2,1,true,false,false>`. Thus this measured official
baseline is an INT8 activation GEMV, not FP16 GEMM. Its codebook formula permits
integer `dp4a` accumulation directly from hashed trellis states; it does not
need our explicit FP4 code lookup and nibble assembly. The captured template
selects the non-residual INT8 variant. Treat accuracy and speed as separate
comparisons; no full-model quality equivalence was established.

```bash
.venv/bin/python sm120_native_test.py --fused --arvq
.venv/bin/python sm120_gemv_test.py
PYTORCH_ALLOC_CONF=expandable_segments:False \
  compute-sanitizer --tool memcheck --error-exitcode 99 \
  .venv/bin/python sm120_gemv_test.py --parity-only
```

## Further decoding, pipelining, and fusion experiments

The current specialized decoder also uses unsigned `dp4a` to sum the hash bytes,
a shared 1 KiB lookup, packed-byte-to-nibble compaction, and explicit prefetch of
the next K64 compressed words before the current MMA instructions. This is a
software prefetch pipeline; no hardware-counter proof of overlap is claimed.
It remains native FP4 tensor-core multiplication.

Best measured complete-projection times after these changes:

| K → N | Specialized, three launches | Single-launch CTA fusion | Official EXL3 |
| --- | ---: | ---: | ---: |
| 6144 → 512 | 10.24 us | 10.25 us | ~8.2 us |
| 512 → 6144 | 9.38 us | 8.22 us | ~6.2 us |
| 6144 → 2048 | 13.56 us | 14.45 us | ~10.3 us |
| 2048 → 6144 | 12.30 us | 14.34 us | ~8.2 us |

**None of these variants beats the measured official baseline.** The specialized
large projection improves substantially over the original 24.6 us four-plane
prototype, but that is a prototype improvement, not an EXL3 speedup.

Two full-fusion implementations are retained as experimental comparison paths:

- `fused_project` uses a cooperative grid with packing, multiplication, and
  reduction separated by grid barriers. It passes parity, but is slower; large
  projections measured roughly 25–28 us at their best tested launch geometry
  before the final decoder changes. Its JSON is labeled separately.
- `FusedGEMV` replicates the necessary input-transform groups within each CTA,
  writes split-K partials, then uses an atomic completion counter per 128-output
  tile. The last CTA performs the output reduction and Hadamard. It requires no
  grid barrier. The workspace is explicit, must be used serially, and must not
  be shared by concurrent invocations. Its returned output buffer is reused.
  Counters are initialized before timing and reset by each successful call.

The output Hadamard uses a separate device function: the initial inlined
cooperative implementation failed parity despite matching packed activations
and MMA partials. That failing form is not retained. The exact cause of that
initial discrepancy has not been established; passing tests do not establish a
compiler bug.

The specialized decoder is bit-identical to the generic FP4 path in the tested
cases. Fusion changes the split-K summation order, so its comparison uses
relative L2 below 1e-5 rather than bit identity. Tests cover nonuniform signed
input/output scales, multiple sizes and split counts, and graph timing.
These experiments do not establish original EXL3 quantization equivalence,
whole-model quality, serving integration, or concurrent workspace safety.

```bash
.venv/bin/python sm120_gemv_test.py
.venv/bin/python sm120_fused_test.py --cta
.venv/bin/python sm120_fused_test.py --cta --parity-only
.venv/bin/python sm120_fused_test.py
```

## Confirmed two-token win (follow-up)

The rate-2 specialized path now supports multiple tokens with two tokens per
MMA tile. It cooperatively loads each compressed word once per warp, broadcasts
it to decoding lanes, and extracts trellis windows with a funnel shift. The
prefetch buffer holds eight assembled windows rather than sixteen source words.
Power-of-two activation scale construction uses exponent bits, and the plane
reduction uses explicit rounded multiplication rather than variable `ldexpf`.
The fused experimental path allocates shared activation storage for its actual
K slice rather than the whole input width.

**The confirmed win is at batch two, not batch one.** Three random seeds with
nonuniform signed scales, alternating benchmark order, and comparison against
the faster of official direct/reconstruction paths give these median latencies:

| K → N, two tokens | Native FP4 | Best official | Throughput ratio |
| --- | ---: | ---: | ---: |
| 6144 → 2048 | 12.296 us | 16.075 us | 1.31x |
| 512 → 6144 | 8.202 us | 10.247 us | 1.25x |

All six comparisons at each shape favor FP4. The measured ranges are
12.291–12.302 versus 15.907–16.344 us for the larger projection, and
8.192–8.222 versus 10.241–10.254 us for the smaller one. These wins use
`project(..., arvq=True, gemv=True)` with splits 32 and 8 respectively, including
input/output Hadamard transforms and activation packing. They use the ordinary
three-launch path, not the experimental whole-projection fusion.

The broader sweep is in `multi_token.json`: batch-one native FP4 still loses;
the 2048→6144 projection at batch two roughly ties. Batch-four results mostly tie
or lose, and batch eight loses. These are shape-specific microbenchmark wins,
not evidence of a faster full model or better quantization accuracy. Native FP4
still approximates the original EXL3 codewords and activations.

Hardware profiling was possible through a privileged `docker exec` of the
isolated benchmark process; no host settings or serving container were changed.
The global-LUT profile identified substantial L1/TEX dependency stalls. The
shared-LUT profile removed that specific recommendation but still showed less
than one full scheduling wave at the tested launch size. Nsight replay durations
are not the warm-graph benchmark durations and must not replace the timing table.

Further experiments retained only as local scratch included a separate-pack
kernel with fused output completion, and an eight-warp full-fusion block. Neither
won. Sweeping 32/64/128/256-thread blocks also did not solve the batch-one gap.

The exponent-bit packer matches commit `a187767` bit-for-bit for all 63,488 finite
FP16 patterns, both ordered and shuffled into groups, and for tested fused input
transforms. Specialized multi-token outputs match the generic FP4 path exactly
in the tested shapes. Full fusion retains its tolerance-based comparison.

```bash
.venv/bin/python sm120_multi_token_test.py
.venv/bin/python sm120_confirm_two.py
.venv/bin/python sm120_multi_token_test.py --parity-only
# Optional packer regression against the prior implementation:
git show a187767:experiments/exl3-speed-v1/native_fp4.cu > local-results/baseline_a187.cu
nvcc -O3 -std=c++17 --shared -Xcompiler=-fPIC \
  -gencode arch=compute_120a,code=sm_120a \
  local-results/baseline_a187.cu -o local-results/baseline_a187.so
.venv/bin/python sm120_pack_regression.py
```
