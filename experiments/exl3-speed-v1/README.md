# EXL3 trellis execution experiments and SM120 handoff

Target: SM120 (RTX PRO 6000 / RTX 50-series). This branch publishes research
kernels, a checkpoint decoder, reproducible benchmark drivers, and measured
B200 results. **There is no production vLLM integration. A native SM120 FP4 research
kernel is now tested; see [SM120 results](SM120_RESULTS.md).** AI-assisted implementation; a human must review before
any upstream contribution. This is a fork research branch, not an upstream PR.

## What changed

The successful MiMo artifacts retain EXL3 mul1 trellises, mixed nominal rates
1.5/2/2.5, input/output scales, and every neuron. Training changed bit allocation
and the objective; kernel experiments do not change the stored codes.

* `trellis_gemm.py`: exact packed-trellis decode through a 1024-entry FP16 table,
  fused decode/FP16 GEMM, experimental FP8-activation/FP4-pair GEMM, split-K
  reduction, fused output Hadamard/scales/scatter, and fused SwiGLU.
* `runtime_probe.py`: official EXL3 direct and reconstruction paths, custom
  paths, and a cached dense BF16 control. `recon_scatter` uses official inner
  reconstruction and GEMM, then our fused output Hadamard/scatter.
* `fp4_pair_probe.py`: weight-only FP4-pair approximation experiment.
* `sm120_fp4_quality.py`: **numerical emulation**, not an SM120 kernel, of
  two FP4 weight planes with one or two FP4 activation planes.
* `decode.py`: reads the packed pilot format and reconstructs original-order
  gate/up/down weights through official EXL3.
* `smoke_test.py`: synthetic packed-code correctness without model data.

The exact mul1 decode hashes a 16-bit trellis state with uint32 multiplication
by `0x83DCD12D`, then sums the four product bytes. Only 1,021 sums are possible.
The original FP16 codebook arithmetic can be represented by a 2 KiB lookup
(table length rounded to 1,024). The approximate table packs two E2M1 indices
into each byte: `weight ~= fp4_hi + fp4_lo / 16`. This table is 1 KiB, shared
by the decoder. The trellis path space and stored weight codes stay unchanged.
This is a simplification of the existing codebook, not a new quantizer theorem.

## Important SM120 distinction

Use SM120 warp-level block-scaled `mma.sync`, not B200 `tcgen05`/TMEM APIs.
The concrete all-FP4 target is E2M1 x E2M1, FP32 accumulation, with native scale
formats. See [NVIDIA's SM120 warp MMA documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html).
For group 32, E8M0 scales and the 16x8x64 MXF4 instruction are a direct starting
point. Group-16 emulation uses power-of-two scales too; a port must check the
chosen hardware scale representation/range rather than assume equivalence.

The first `gemm_fp4` Triton shape (BM16/BN64) compiled on B200 into BF16 MMA,
**not native FP4**. Increasing BM to 128 failed in Triton's compiler pass.
Those attempts are not evidence of native FP4 performance or an SM120 limit.
`fp4` / `fp4_residual` / `fp4_scatter` currently use FP8 activations and FP4
weights. They are retained as tested prototypes, not the proposed all-FP4 port.
`fp4_scatter` includes the FP8 activation residual (four MMA terms).

## Quality evidence

Four development pilots; native reference is the released **MXFP4 checkpoint
decoded and evaluated in BF16**, not unavailable original BF16 weights.
Metric: route-weighted relative expert-output RMS. The audit split has been
inspected during research. No whole-model KL or fresh qualification is implied.

| Expert | Original packed val / audit | All-FP4 emulation val / audit |
| --- | --- | --- |
| 21/137 | 5.968% / 5.683% | 6.048% / 5.817% |
| 21/32 | 6.562% / 5.471% | 6.733% / 5.661% |
| 21/201 | 9.558% / 7.442% | 9.657% / 7.555% |
| 69/19 | 9.357% / 9.456% | 9.453% / 9.564% |

Right column: group 32, two independently scaled FP4 activation planes,
two FP4 weight planes, four product terms. This emulates the operand values
with FP32 products/sums on B200; a real kernel's rounding must be measured.
One activation plane (two terms) failed <10% on three of four experts.
Group-16 two-plane emulation also passed all four.

Artifacts are 1.99572–1.99832 actual bits/original weight, including pilot
headers, scale arrays and block metadata. Decoder tables are global constants.
Runtime workspace and expanded dense controls are not persistent model bytes;
the benchmark keeps reference weights resident and is not a VRAM-footprint test.
`results/pilot_manifest.json` records artifact hashes; model/capture files are
not embedded in git.

## Timing evidence (B200 only)

Whole single-expert gate/up/SwiGLU/down, warm CUDA graphs, no routing or
communication. Official direct, official reconstruct-each-call, and cached
BF16 are different baselines: compare against the **best official path** at
each batch size, not only forced-direct at larger batches.

Initial custom fused FP16 and FP4 paths are slower than the best official path.
The fused **reconstruction** epilogue did improve the matched reconstruction
baseline. Batch 1 microseconds, custom / comparison in that same run:

| Expert | `recon_scatter` | reconstruct + separate epilogue |
| --- | ---: | ---: |
| 21/137 | 81.9 | 94.3 |
| 21/32 | 79.9 | 90.3 |
| 21/201 | 69.6 | 77.8 |
| 69/19 | 75.8 | 84.0 |

These are prototype microbenchmarks, not an SM120 claim, ARVQ comparison, or
end-to-end serving gain. The comparison in scatter runs also uses fused SwiGLU
and explicit output row maps. Earlier official-only runs differ slightly in
assembly; use the published driver to remeasure on the same hardware.

Raw JSON is in `results/`. Historical files without `block_m`/`block_n` used
BM16/BN64. New runs include these settings in both JSON and filenames to avoid
overwriting measurements. Historical `runtime_layer*.json` uses
`native_exl3_us`; later runs use `compressed_path_us`.

## Reproduce

Run from this directory in the checkout. Requires a CUDA GPU, torch, Triton,
numpy, safetensors, and official EXL3. Tested environment:
`torch 2.11.0+cu130`, official `exllamav3 1.5.1+cu128.torch2.11.0`, CUDA12
runtime/cuBLAS libraries for that wheel. Match the wheel to your torch/CUDA;
these scripts do not install or alter vLLM.

On the existing B200 host, SSH port 2222, `coder-jarrel-jarrel-b200`:

```bash
export PY=/data/jarrel/venv-exl3-baseline/bin/python
export LD_LIBRARY_PATH=/data/jarrel/venv-exl3-baseline/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/data/jarrel/venv-exl3-baseline/lib/python3.12/site-packages/nvidia/cublas/lib
export OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "$PY" smoke_test.py
CUDA_VISIBLE_DEVICES=0 "$PY" runtime_probe.py --index 0 --backend official --splits 8
CUDA_VISIBLE_DEVICES=0 "$PY" runtime_probe.py --index 0 --backend recon_scatter --splits 8
CUDA_VISIBLE_DEVICES=0 "$PY" runtime_probe.py --index 0 --backend fused_scatter --splits 8
CUDA_VISIBLE_DEVICES=0 "$PY" sm120_fp4_quality.py --index 0 --group 32
```

Use indices 0–3 for the four experts. Paths can be overridden:

* `MIMO_ARTIFACT_ROOT`: defaults `/data/jarrel/mimo-known-baselines-v1`.
  Relative paths are in `common.py:CANDIDATES`; each ends in `selected.bin`.
* `MIMO_WORK_ROOT`: defaults `/data/jarrel/mimo-v26-arvq-hot5`.
  Needs `source/` (native safetensors + config/index) and
  `capture21/rank{0..7}.pt`, `capture69/rank{0..7}.pt`.
* `MIMO_RESULT_ROOT`: defaults `local-results/` next to scripts, git-ignored.
* `MIMO_BM` and `MIMO_BN`: experimental tile sizes, default 16 and 64.
  N must be divisible by BN; current kernels are for aligned pilot shapes.

The four packed files alone total about 38 MB. To benchmark on SM120 with real
quality checks, also transfer the required native expert shards and captures;
copying only the packed files is enough to decode, not to evaluate error.
The synthetic smoke test needs none of those files.

## Next work for the SM120 agent

1. Run the synthetic test and official single-expert baselines on actual SM120.
2. Implement **two independently scaled FP4 activation planes** and the two
   weight planes generated from the trellis lookup. Accumulate four FP4 MMA
   terms. Confirm native instructions in PTX/SASS; do not infer from API names.
3. Keep decode and operand packing in registers/shared memory; avoid persisting
   expanded weights. Reduce duplicate bit extraction/lookup across fragment
   lanes. Preserve exact wraparound and fractional 1.5/2.5-bit indexing.
4. Compare fused-decode vs reconstruct-each-call, with fused Hadamard/scatter
   and SwiGLU. Dispatch by batch size only after measurement. Report workspace,
   actual stored bytes, and both official baseline paths.
5. Verify actual kernel output on all four experts. Then add routed/grouped MoE
   batching and vLLM integration. Current Python prototype launches per part;
   mixed-rate part scheduling is a remaining overhead.
6. Compare to ARVQ on the **same SM120 GPU** and token/expert shapes. There is no
   matched ARVQ throughput result in this handoff.

## Quantization lessons and full-model follow-up

Mixed projection/block rates, output sensitivity through SwiGLU/down, and joint
scale fitting are the tested quality improvements. Some experts favor fewer
up bits; the difficult 21/201 favored fewer gate bits and more down bits. Do not
hardcode one allocation across the model. Preserve every neuron. Fit selection
using training/development data, then qualify on fresh data and model KL.

MiMo full-model conversion has been requested but **has not been launched by
this kernel handoff**. There are 69 routed layers x 384 experts = 26,496 experts.
The four pilot recipes are not yet a resumable full-model converter. Existing
captures inspected at the work root cover layers 1, 2 and 21–69; complete
coverage and sufficient routed samples must be checked before batch conversion.
Non-expert attention, dense, audio and vision weights need an explicit policy;
<2 bpw on four routed experts does not mean the whole checkpoint averages 2 bpw.

The user also requested investigating **GLM 5.3** versus the existing ARVQ
system. Reuse the decoder/kernel design, but refit GLM calibration, scales and
allocation. Start with spread layers/experts at equal serialized byte budget;
compare output error and fresh held-out model KL, then SM120 latency/memory.
MiMo pilot gains do not establish a GLM improvement. Do not overwrite either
existing checkpoint while exploring.

## Handoff validation

After removing temporary-script dependencies and formatting the published files:

* `ruff check` and Python byte compilation passed.
* Synthetic B200 test passed exact reconstructed symbols at 1.5/2/2.5 bits and
  GEMM/Hadamard/scatter checks at batches 1, 4, 16, 64.
* Eight B200 jobs reran the packaged entry points: all four group-32 FP4 operand
  quality tests reproduced their earlier results exactly; official,
  `recon_scatter`, `fused_scatter`, and `fp4_scatter` completed expert 21/201
  quality and timing. JSON is in `results/handoff-verification/`.

Current user priority: **MiMo v2.6 Pro first**. GLM 5.3 is a later comparison.

## Locally fitted FP4 codebooks

See [weight component support](WEIGHT_COMPONENTS.md) for shared or per-expert
codebooks with two or four FP4 weight components, API examples and validation
status. Four-component support is experimental pending native SM120 checks.
