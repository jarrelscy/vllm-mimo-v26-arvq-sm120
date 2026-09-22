# Pinned fitting recipe

`provenance.json` records the **original** SHA256 of files copied from the
existing `/home/coder/git/glm52/btx53` ARVQ implementation. It is not a hash
manifest of the adapted files. Changes for this campaign:

- Extract the existing `Projection` without GLM-specific model imports.
- Broadcast boundary row weights along with the expert-parallel training batch.
- Derive the expert count from configuration (384 for MiMo).
- Use normal layer-numbered output paths, including layer 3.
- Label the objective and validation distribution accurately for this capture.
- Keep initial-fit Hessian factors/diagonals in FP32. Solve triangular feedback
  systems directly instead of materializing an inverse of each diagonal block.

The continuous optimizer, discrete output-gradient proposals, separate training
acceptance check, activation-plane emulation and validation selection remain the
existing algorithm. Vendor sources are excluded from mechanical lint rewrites;
campaign adapters and new tests are linted normally.
