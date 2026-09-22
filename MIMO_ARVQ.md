# MiMo-V2.6-Pro-RL ARVQ campaign

Work in progress. No MiMo ARVQ serving or quality claim yet.

This fork starts from `jarrelscy/vllm-glm52-sm120` commit
`d7ada6d5e5dfd420d262a6bca93a0d19361edecd`. It retains the existing
per-expert ARVQ v4/v5 kernels while adding a separate MiMo campaign.
The GLM launchers and format integration are not MiMo launch commands.

Initial checks: the calibration source reader passed known-code, block-scale,
and invalid-input tests. A real layer-1 expert decoded all three projections
with the expected shapes and finite values. Native MiMo chat formatting and
tokenizer round-trip passed with a separate reasoning_content field. These
checks do not yet establish serving-kernel parity or quantized model quality.

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

MiMo's H=6144 and TP4 intermediate=512 satisfy the existing ARVQ loader's
multiple-of-128 shape requirement. Its hyb_kind array is sized dynamically
from num_experts, so the 384-expert roster is not inherently limited to GLM's
256. Routing, expert-id handling and all CUDA dispatch paths still need a real
384-expert TP4 test; shape eligibility alone does not qualify the kernels.

## Running campaign

The B200 work directory is `/data/jarrel/mimo-v26-arvq`. `driver.py` sequences
layers 1–69: capture, per-expert Hessian initial fit, immutable initial export,
PV, accepted export, and propagation of the selected weights into next-layer
training/validation/audit inputs. The first dense layer is retained unchanged.
`publish.py` runs independently and replaces each initial layer with its accepted
PV export using one HF commit for weights, manifests and the tensor index.
The backbone and all auxiliary modules are retained from the pinned source.

Training traverses 18,006,461 native-tokenized text tokens without replacement,
up to 69 updates of 262,144 tokens, split into four 65,536-token microbatches.
Each of eight B200 ranks owns 48 experts. The initial fit uses representative
activation samples; PV uses the full corpus. Books are individual per expert,
constrained to FP4; scales are FP16 per row/block of 128 weights (format v4).
Adam starts at 0.048 for books and 0.032 for log-scales, with the existing
validation-based early reduction and stopping rule. Discrete reassignment runs
every ten updates. Held-out validation selects checkpoints every five updates;
initialization remains eligible. A separate audit checks the single selected
checkpoint and can retain initialization if that checkpoint regresses.

The fixed validation and audit sets each contain 16K tokens across all six
corpus categories, separated by exact document identity from training. This
does not establish semantic deduplication across related documents. No image
or audio embeddings are used for fitting. An additional 16K-token document
set, unused by fitting or checkpoint selection, is reserved for end-to-end
teacher/student likelihood and KL measurements after all layers complete.

The objective is routed MoE output error on the **same student inputs**, with
native released MXFP4 experts as targets. FP4 activation planes and FP16
nonlinear boundaries are emulated; actual serving accumulation is not executed.
Attention/dense FP8 weights are decoded for BF16 reference computation. Thus
even the final full-model evaluation is an emulation, and does not replace
the SM120 serving and multimodal gates above.

`pipeline_status.json`, `logs/`, `receipts/`, per-layer `report.json` and
`upload_state.json` provide local state. The HF `pv_progress.json` records only
verified uploads. Failed stages retry up to three times; PV retries resume the
last complete optimizer checkpoint. Incomplete exports are never published.
`STOP` requests a stop at the next layer boundary; remove it before restarting.

The first layer-1 trial was invalidated after detecting that its capture rounded
the FP32 router correction bias to BF16. Its captures, fits and propagation are
archived under `invalidated_router_bias`; its reported improvement is not a
production result. The replacement campaign preserves the native router bias
and requires explicit dtype/top-k parity qualification before starting.
Propagation now shares input-plane quantization across experts; real-weight
checks found bitwise-identical expert outputs.

Additional serving work remains: the Omni wrapper now accepts
`mimo_qkv_layout="grouped"` to select the Pro language backbone, but the HF
architecture override has not been serving-qualified. The existing MTP loader
still assumes simple TP slicing of grouped QKV tensors, which requires review
for TP4. MTP output projections already bypass quantization and retain BF16.
All three embedded MTP layers are preserved, while the inherited serving code
currently activates only the first. These are deployment gates, not changes to
the text fitting objective.

## Approved 5% hot transition

The user approved 1325 NVFP4 hot experts out of 26496 routed experts (5.0008%).
`queue_hybrid.py` waits for the current candidate campaign and its evaluation,
then runs `prepare_hybrid.py` and a second sequential PV campaign in
`/data/jarrel/mimo-v26-arvq-hot5`. It does not wait for successful HF uploads.
The existing fits are warm starts, not discarded work.

Allocation globally ranks routing-weighted output-error reduction from NVFP4
versus the accepted ARVQ candidates on retained training probes. Each layer is
capped at 192 hot experts; the total remains 1325. This is an additive expert
error proxy, measured on text only, not full-model loss. Native released MXFP4
experts supply the reference; hot weights are converted to E2M1 with E4M3 scales
per 16 weights and FP32 projection globals. Packed hot weights are used in both
frozen-output capture and propagation. Validation/audit tokens do not choose
allocation. The full-model test evaluates the final hybrid separately.

The hybrid pass recaptures student inputs from layer 1, subtracts frozen hot
output from the PV training target, and tunes only the remaining cold experts.
The previous Adam/index/early-stopping settings remain unchanged. The publisher
commits weights, hot/cold roster, config and tensor index atomically per layer.
Both campaigns share a publisher lock to prevent stale concurrent overwrites.

The 5% estimate is 350.9 GiB for weights plus 1M BF16 main-model KV, including
replicated expert books. Runtime allowance of 20–30 GiB gives 370.9–380.9 GiB;
actual TP4 usable memory, auxiliary replication, speculative KV and workspace
usage still require measurement. Allocation is approved; serving fit is not yet
qualified. The previous all-cold completion ETA is not a final-hybrid ETA.
