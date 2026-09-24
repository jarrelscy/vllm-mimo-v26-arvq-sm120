#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p local-results
"${NVCC:-nvcc}" -O3 -std=c++17 --shared -Xcompiler=-fPIC \
  -gencode arch=compute_120a,code=sm_120a -lineinfo \
  native_fp4.cu -o local-results/native_fp4.so
