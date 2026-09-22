# MiMo-V2.6-Pro-RL ARVQ campaign

Work in progress. No MiMo ARVQ serving or quality claim yet.

This fork starts from `jarrelscy/vllm-glm52-sm120` commit
`d7ada6d5e5dfd420d262a6bca93a0d19361edecd`. It retains the existing
per-expert ARVQ v4/v5 kernels while adding a separate MiMo campaign.
The GLM launchers and format integration are not MiMo launch commands.

Target: four RTX PRO 6000 Blackwell GPUs, one 1,048,576-token sequence,
with vision, audio, audio tokenizer, embedded MTP and five-layer DFlash
preserved. Output: `jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid`.

## Preparation

`tools/mimo_arvq/stage_source.py --work WORK` downloads the entire pinned
release (including subdirectories), then checks every file size and all
published LFS SHA256 hashes. The source is never overwritten by fitting.
The script records source_manifest.json, source_status.json and, only on
success, source_verified.json. Re-running resumes Hugging Face downloads.

`tools/mimo_arvq/campaign.json` records the target and outstanding gates.
It is a preparation manifest, not an executable training configuration.

## Port and fitting order

1. Verify source MXFP4 unpacking, scale semantics and all auxiliary modules.
2. Build native MiMo tokenization from raw text/messages, preserving document
   identities and disjoint train/validation/audit sets. Never reuse GLM token
   IDs or captured hidden states. Include prose, code, reasoning, tool traces,
   and multimodal examples; do not copy GLM's token IDs or 50x weighting.
3. Qualify the reference forward and ARVQ expert reconstruction on early,
   middle and late MoE layers. Use per-expert books, FP16 block scales,
   gradient optimization and activation-weighted discrete reassignment.
4. Compute actual ARVQ-vs-hot output-error allocation scores. Choose the hot
   budget from measured TP4 memory with the complete auxiliary modules.
5. Fit sequentially with rolling student inputs, same-input reference targets,
   held-out checkpoint selection and early stopping. Export and validate
   indices, codebooks, scales and allocation together before publishing.
6. Check full-model held-out PPL/KLD, reasoning termination, vision/audio,
   speculative acceptance and long-context quality against the source.

## Initial memory arithmetic (GiB across four GPUs)

The 69 MoE layers contain 1,000,190,509,056 routed weights. All-cold
ARVQ requires 247.43 GiB for indices and FP16 block scales, plus approximately
0.1 GiB of expert books. The format is 2.125 bpw, not 2.0 bpw including scales.
Main-checkpoint non-expert tensors total 32.30 GiB. Conservatively also retain
the separate DFlash (5.16 GiB) and audio tokenizer (1.74 GiB).

Ten global layers, eight KV heads, K=192 and V=128 need 50 GiB of BF16 KV
at 1,048,576 tokens, or 25 GiB with FP8 KV. Sliding-window storage, speculative
buffers, encoder activations, replication, allocator padding and workspaces
are additional. Provisional total with 20–30 GiB runtime allowance is
357–367 GiB (BF16 KV) or 332–342 GiB (FP8 KV). Each 10% of routed weights
promoted to NVFP4 adds about 27.65 GiB. These are estimates, not measurements.

Current local Triton DiffKV fallback has no FP8 KV support. MiMo's K/V shapes,
attention sinks, sliding-window eviction, multimodal class dispatch and
DFlash integration must all be tested on SM120. A text-only successful load
does not satisfy the target. Do not publish production_ready=true until the
entire serving gate passes.
