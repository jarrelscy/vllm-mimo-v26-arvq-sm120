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
