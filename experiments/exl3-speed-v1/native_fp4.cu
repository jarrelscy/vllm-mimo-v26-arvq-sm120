// SPDX-License-Identifier: Apache-2.0
// Experimental EXL3 mul1 -> two FP4 planes, native SM120 MXFP4 MMA.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ void mma(float* d, const unsigned* a, unsigned b0,
                                    unsigned b1, unsigned sa, unsigned sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, "
      "{0,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa),
        "r"(sb));
}

template <bool HAD = false>
__global__ void pack_x(const half* x, unsigned* out, unsigned char* scales,
                       int K, int M, const half* su = nullptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * K) return;
  int row = i / K, k = i % K;
  float value = __half2float(x[i]);
  if constexpr (HAD) {
    __shared__ float tmp[128];
    value = __half2float(__hmul(x[i], su[k]));
    for (int stride = 1; stride <= 16; stride *= 2) {
      float other = __shfl_xor_sync(0xffffffff, value, stride);
      value = (threadIdx.x & stride) ? other - value : value + other;
    }
    for (int stride = 32; stride <= 64; stride *= 2) {
      tmp[threadIdx.x] = value;
      __syncthreads();
      float other = tmp[threadIdx.x ^ stride];
      __syncthreads();
      value = (threadIdx.x & stride) ? other - value : value + other;
    }
    value = __half2float(__float2half_rn(value * 0.08838834764831845f));
  }
  const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int p = 0; p < 2; ++p) {
    float peak = fabsf(value);
    for (int delta = 16; delta; delta /= 2)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, delta));
    int e = (int)ceilf(log2f(fmaxf(peak / 6, 0x1p-24f)));
    float scale = exp2f((float)e), v = fabsf(value) / scale;
    unsigned q = (v > .25f) + (v > .75f) + (v > 1.25f) + (v > 1.75f) +
                 (v > 2.5f) + (v > 3.5f) + (v > 5.f);
    q |= value < 0 ? 8 : 0;
    value -= levels[q & 7] * scale * (q & 8 ? -1.f : 1.f);
    unsigned bits = q << (4 * (k & 7));
    for (int delta = 4; delta; delta /= 2)
      bits |= __shfl_xor_sync(0xffffffff, bits, delta);
    if (!(k & 7)) out[((row * 2 + p) * K + k) / 8] = bits;
    if (!(k & 31)) scales[((row * 2 + p) * K + k) / 32] = e + 127;
  }
}

template <int KA, int HALF>
__device__ __forceinline__ unsigned symbol(const unsigned* packed,
                                           const unsigned char* lut, int k,
                                           int n, int N) {
  int r = k & 15, c = n & 15;
  int lane = (c % 8) * 4 + (r % 8) / 2;
  int pos = lane * 8 + r % 2 + 2 * (r / 8) + 4 * (c / 8);
  constexpr int bits = 256 * KA + 128 * HALF, words = bits / 32;
  int end = (pos + 1) * KA + (pos + 1) / 2 * HALF + bits;
  const unsigned* tile = packed + ((k / 16) * (N / 16) + n / 16) * words;
  unsigned a = tile[((end - 16) / 32) % words];
  unsigned b = tile[((end - 1) / 32) % words];
  unsigned state =
      end % 32 == 0
          ? b & 65535u
          : (unsigned)((((uint64_t)a << 32) | b) >> (32 - end % 32)) & 65535u;
  unsigned v = state * 0x83DCD12Du;
  return __ldg(lut + (v & 255) + ((v >> 8) & 255) + ((v >> 16) & 255) +
               (v >> 24));
}

template <int KA, int HALF>
__global__ void project(const unsigned* packed, const unsigned char* table,
                        const unsigned* x, const unsigned char* xs,
                        float* partial, int M, int N, int K, int S) {
  const unsigned char* lut = table;
  int lane = threadIdx.x % 32, q = lane / 4, c = lane % 4;
  int tile = blockIdx.x * 4 + threadIdx.x / 32;
  if (tile >= N / 16) return;
  int token = blockIdx.z * 8 + q, G = K / 64;
  float d[4] = {};
  for (int g = G * blockIdx.y / S; g < G * (blockIdx.y + 1) / S; ++g) {
    unsigned a[4] = {}, b[4] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int n = tile * 16 + q + (j % 2) * 8;
      int k = g * 64 + c * 8 + (j / 2) * 32;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        unsigned v = symbol<KA, HALF>(packed, lut, k + t, n, N);
        a[j] |= (v & 15) << (t * 4);
        b[j] |= (v >> 4) << (t * 4);
      }
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      int base = ((token * 2 + p) * K + g * 64) / 8;
      unsigned b0 = token < M ? x[base + c] : 0;
      unsigned b1 = token < M ? x[base + c + 4] : 0;
      int si = ((token * 2 + p) * K + g * 64) / 32;
      unsigned sb = token < M ? xs[si] | (unsigned(xs[si + 1]) << 8) : 0x7f7f;
      mma(d, a, b0, b1, 0x7f7f7f7f, sb);
      mma(d, b, b0, b1, 0x7b7b7b7b, sb);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int m = blockIdx.z * 8 + c * 2 + j % 2;
    int n = tile * 16 + q + (j / 2) * 8;
    if (m < M) partial[(blockIdx.y * M + m) * N + n] = d[j];
  }
}

extern "C" int exl3_pack(const void* x, void* out, void* scales, int M, int K,
                         void* stream) {
  if (M < 1 || K < 64 || K % 64) return cudaErrorInvalidValue;
  pack_x<false><<<(M * K + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const half*)x, (unsigned*)out, (unsigned char*)scales, K, M);
  return cudaGetLastError();
}

extern "C" int exl3_project(const void* packed, const void* lut, const void* x,
                            const void* xs, void* partial, int M, int N, int K,
                            int S, int rate2, void* stream) {
  if (M < 1 || N < 16 || N % 16 || K < 64 || K % 64 || S < 1 || S > K / 64)
    return cudaErrorInvalidValue;
  dim3 grid((N + 63) / 64, S, (M + 7) / 8);
#define LAUNCH(A, B)                                                          \
  project<A, B><<<grid, 128, 0, (cudaStream_t)stream>>>(                      \
      (const unsigned*)packed, (const unsigned char*)lut, (const unsigned*)x, \
      (const unsigned char*)xs, (float*)partial, M, N, K, S)
  if (rate2 == 3) {
    LAUNCH(1, 1);
  } else if (rate2 == 4) {
    LAUNCH(2, 0);
  } else if (rate2 == 5) {
    LAUNCH(2, 1);
  } else
    return cudaErrorInvalidValue;
  return cudaGetLastError();
}

extern "C" int exl3_had_pack(const void* x, const void* su, void* out,
                             void* scales, int M, int K, void* stream) {
  if (M < 1 || K < 128 || K % 128) return cudaErrorInvalidValue;
  pack_x<true><<<M * K / 128, 128, 0, (cudaStream_t)stream>>>(
      (const half*)x, (unsigned*)out, (unsigned char*)scales, K, M,
      (const half*)su);
  return cudaGetLastError();
}
