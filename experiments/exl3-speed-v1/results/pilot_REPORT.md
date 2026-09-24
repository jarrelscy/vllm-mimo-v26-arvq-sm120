# Verified MiMo two-bit pilot results

All four pilot experts pass relative output RMS <10% on both development validation and audit at no more than 2 bits per original weight, retaining every neuron. Verified twice in fresh processes with identical results.

| Expert | Validation | Audit | Actual bpw | Packed bytes |
| --- | ---: | ---: | ---: | ---: |
| 21/137 | 5.9681% | 5.6830% | 1.997494592 | 9,425,362 |
| 21/201 | 9.5577% | 7.4424% | 1.998323229 | 9,429,272 |
| 21/32 | 6.5619% | 5.4713% | 1.995720334 | 9,416,990 |
| 69/19 | 9.3569% | 9.4559% | 1.997494592 | 9,425,362 |

Each expert has 37,748,736 original weights and 2,048 intermediate neurons. The byte counts include the 4,096-byte header, packed neuron mask, trellis codes, all FP16 scales, projection/block metadata and codebook markers. All output rows are covered exactly once. SHA-256 hashes and raw error/reference norms are recorded in manifest.json and the two repeat reports.

The selected .bin files are copied locally alongside this report. Their hashes match the freshly verified remote originals. decode.py reconstructs gate/up/down matrices in original ordering using official EXL3 1.5.1; these are research artifacts, not a production inference integration.

Expert 201 uses output-sensitive EXL3 quantization for gate/up, calibrated input covariance, damping 0.1, mixed nominal rates 1.5/2/2.5 with a down segment reduced to 2 bits, then joint scale fitting. This is an application and adaptation of established trellis methods, not a claimed new mathematical theorem. The other experts use measured mixed/block bit allocation and joint scale fitting.

Reference: decoded native MXFP4 weights evaluated with BF16 expert computation. This proves the four requested pilot metrics, not whole-model quality, KL or perplexity. Audit was inspected repeatedly during development; it is not a fresh qualification dataset. No full model was uploaded.

Reproduction environment on coder-jarrel-jarrel-b200: /data/jarrel/venv-exl3-baseline/bin/python with EXL3 1.5.1+cu128.torch2.11.0 and the CUDA12 runtime directory in LD_LIBRARY_PATH. Source weights and captures are under /data/jarrel/mimo-v26-arvq-hot5. The experiment repository contains experiments/known-baselines-v1/verify_final.py; run indices 0–3, with independent repeats 0 and 1.

Baseline and failed-experiment evidence is retained in /home/coder/git/vllm-mimo-v26-arvq-sm120/experiments/known-baselines-v1/results/.
