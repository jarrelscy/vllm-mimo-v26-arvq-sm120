// SPDX-License-Identifier: Apache-2.0
// Experimental EXL3 mul1 -> two FP4 planes, native SM120 MXFP4 MMA.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cooperative_groups.h>

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

__device__ __forceinline__ void mma_arvq(float* d, const unsigned* a,
                                         unsigned b0, unsigned b1, unsigned sa,
                                         unsigned sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, "
      "{0,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa),
        "r"(sb));
}

template <bool HAD = false, bool ARVQ = false>
__device__ void pack_body(const half* x, unsigned* out, unsigned char* scales,
                          int K, int M, const half* su, int bx) {
  int i = bx * blockDim.x + threadIdx.x;
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
  constexpr int P = ARVQ ? 4 : 2;
  constexpr int GROUP = ARVQ ? 16 : 32;
  for (int p = 0; p < P; ++p) {
    float peak = fabsf(value);
    for (int delta = GROUP / 2; delta; delta /= 2)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, delta));
    int e = (int)ceilf(log2f(fmaxf(peak / 6, 0x1p-24f)));
    if constexpr (ARVQ) e = max(-6, min(8, e));
    float scale = exp2f((float)e), v = fabsf(value) / scale;
    unsigned q = (v > .25f) + (v > .75f) + (v > 1.25f) + (v > 1.75f) +
                 (v > 2.5f) + (v > 3.5f) + (v > 5.f);
    q |= value < 0 ? 8 : 0;
    value -= levels[q & 7] * scale * (q & 8 ? -1.f : 1.f);
    if constexpr (ARVQ) value *= 16;
    unsigned bits = q << (4 * (k & 7));
    for (int delta = 4; delta; delta /= 2)
      bits |= __shfl_xor_sync(0xffffffff, bits, delta);
    if (!(k & 7)) out[((row * P + p) * K + k) / 8] = bits;
    if (!(k % GROUP))
      scales[((row * P + p) * K + k) / GROUP] = ARVQ ? ((e + 7) << 3) : e + 127;
  }
}

template <bool HAD = false, bool ARVQ = false>
__global__ void pack_x(const half* x, unsigned* out, unsigned char* scales,
                       int K, int M, const half* su = nullptr) {
  pack_body<HAD, ARVQ>(x, out, scales, K, M, su, blockIdx.x);
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

__device__ __forceinline__ unsigned compact_nibbles(unsigned x) {
  x &= 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  return (x | (x >> 8)) & 0xffffu;
}

__device__ __forceinline__ void prefetch_words(const unsigned* packed,
                                               unsigned* left, unsigned* right,
                                               int g, int c, int q, int tile,
                                               int N) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int k = g * 64 + c * 8 + h * 32;
#pragma unroll
    for (int t = 0; t < 8; t += 2) {
      int r = (k + t) & 15;
      int pos = (q * 4 + (r % 8) / 2) * 8 + 2 * (r / 8);
      int end = (pos + 6) * 2;
      const unsigned* src = packed + (((k + t) / 16) * (N / 16) + tile) * 16;
      left[h * 4 + t / 2] = src[((end - 26 + 512) / 32) & 15];
      right[h * 4 + t / 2] = src[((end - 1) / 32) & 15];
    }
  }
}

template <int KA, int HALF, bool ARVQ = false, bool GEMV = false>
__device__ void project_body(const unsigned* packed, const unsigned char* table,
                             const unsigned* x, const unsigned char* xs,
                             float* partial, int M, int N, int K, int S, int bx,
                             int by, int bz) {
  const unsigned char* lut = table;
  int lane = threadIdx.x % 32, q = lane / 4, c = lane % 4;
  int tile = bx * 4 + threadIdx.x / 32;
  if (tile >= N / 16) return;
  int token = bz * 8 + q, G = K / 64;
  float d[4] = {};
  unsigned ahead_left[8], ahead_right[8];
  if constexpr (GEMV)
    prefetch_words(packed, ahead_left, ahead_right, G * by / S, c, q, tile, N);
  for (int g = G * by / S; g < G * (by + 1) / S; ++g) {
    unsigned a[4] = {}, b[4] = {};
    if constexpr (GEMV) {
      // Four neighboring trellis states feed both halves of the MMA row tile.
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int k = g * 64 + c * 8 + h * 32;
#pragma unroll
        for (int t = 0; t < 8; t += 2) {
          int r = (k + t) & 15;
          int pos = (q * 4 + (r % 8) / 2) * 8 + 2 * (r / 8);
          int end = (pos + 6) * 2;
          unsigned left = ahead_left[h * 4 + t / 2];
          unsigned right = ahead_right[h * 4 + t / 2];
          unsigned window = end % 32 == 0
                                ? right
                                : (unsigned)((((uint64_t)left << 32) | right) >>
                                             (32 - end % 32));
#pragma unroll
          for (int row = 0; row < 2; ++row) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              unsigned hash =
                  ((window >> (10 - row * 8 - u * 2)) & 65535u) * 0x83DCD12Du;
              unsigned v = lut[__dp4a(hash, 0x01010101u, 0u)];
              if (t < 4)
                a[h * 2 + row] |= v << ((t + u) * 8);
              else
                b[h * 2 + row] |= v << ((t + u - 4) * 8);
            }
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned low = a[j], high = b[j];
        a[j] = compact_nibbles(low) | (compact_nibbles(high) << 16);
        b[j] = compact_nibbles(low >> 4) | (compact_nibbles(high >> 4) << 16);
      }
    } else {
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
    }
    if constexpr (GEMV) {
      if (g + 1 < G * (by + 1) / S)
        prefetch_words(packed, ahead_left, ahead_right, g + 1, c, q, tile, N);
    }
    if constexpr (ARVQ) {
      int plane_slot = bz * 8 + q;
      int base = (plane_slot * K + g * 64) / 8;
      bool valid = plane_slot < M * 4;
      unsigned b0 = valid ? x[base + c] : 0;
      unsigned b1 = valid ? x[base + c + 4] : 0;
      unsigned sb =
          valid ? reinterpret_cast<const unsigned*>(xs)[plane_slot * G + g]
                : 0x38383838u;
      mma_arvq(d, a, b0, b1, 0x38383838u, sb);
      mma_arvq(d, b, b0, b1, 0x18181818u, sb);
    } else {
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
  }
  if constexpr (ARVQ) {
    // Each pair of lanes covers the four residual planes of one token.
    float lo = ldexpf(d[0], -8 * (c % 2)) + ldexpf(d[1], -8 * (c % 2) - 4);
    float hi = ldexpf(d[2], -8 * (c % 2)) + ldexpf(d[3], -8 * (c % 2) - 4);
    lo += __shfl_xor_sync(0xffffffff, lo, 1);
    hi += __shfl_xor_sync(0xffffffff, hi, 1);
    int m = bz * 2 + c / 2;
    if (!(c & 1) && m < M) {
      partial[(by * M + m) * N + tile * 16 + q] = lo;
      partial[(by * M + m) * N + tile * 16 + q + 8] = hi;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int m = bz * 8 + c * 2 + j % 2;
      int n = tile * 16 + q + (j / 2) * 8;
      if (m < M) partial[(by * M + m) * N + n] = d[j];
    }
  }
}

template <int KA, int HALF, bool ARVQ = false, bool GEMV = false>
__global__ void project(const unsigned* packed, const unsigned char* table,
                        const unsigned* x, const unsigned char* xs,
                        float* partial, int M, int N, int K, int S) {
  __shared__ unsigned char local_lut[1024];
  if constexpr (GEMV) {
    reinterpret_cast<unsigned long long*>(local_lut)[threadIdx.x] =
        reinterpret_cast<const unsigned long long*>(table)[threadIdx.x];
    __syncthreads();
  }
  project_body<KA, HALF, ARVQ, GEMV>(packed, GEMV ? local_lut : table, x, xs,
                                     partial, M, N, K, S, blockIdx.x,
                                     blockIdx.y, blockIdx.z);
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

extern "C" int exl3_arvq_pack(const void* x, const void* su, void* out,
                              void* scales, int M, int K, void* stream) {
  if (M < 1 || K < 128 || K % 128) return cudaErrorInvalidValue;
  if (su) {
    pack_x<true, true><<<M * K / 128, 128, 0, (cudaStream_t)stream>>>(
        (const half*)x, (unsigned*)out, (unsigned char*)scales, K, M,
        (const half*)su);
  } else {
    pack_x<false, true><<<(M * K + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const half*)x, (unsigned*)out, (unsigned char*)scales, K, M);
  }
  return cudaGetLastError();
}
extern "C" int exl3_arvq_project(const void* packed, const void* lut,
                                 const void* x, const void* xs, void* partial,
                                 int M, int N, int K, int S, int rate2,
                                 void* stream) {
  if (M < 1 || N < 16 || N % 16 || K < 128 || K % 128 || S < 1 || S > K / 64)
    return cudaErrorInvalidValue;
  dim3 grid((N + 63) / 64, S, (M + 1) / 2);
#define LAUNCH_ARVQ(A, B)                                                     \
  project<A, B, true><<<grid, 128, 0, (cudaStream_t)stream>>>(                \
      (const unsigned*)packed, (const unsigned char*)lut, (const unsigned*)x, \
      (const unsigned char*)xs, (float*)partial, M, N, K, S)
  if (rate2 == 3) {
    LAUNCH_ARVQ(1, 1);
  } else if (rate2 == 4) {
    LAUNCH_ARVQ(2, 0);
  } else if (rate2 == 5) {
    LAUNCH_ARVQ(2, 1);
  } else
    return cudaErrorInvalidValue;
  return cudaGetLastError();
}

extern "C" int exl3_fp4_gemv(const void* packed, const void* lut, const void* x,
                             const void* xs, void* partial, int M, int N, int K,
                             int S, int rate2, void* stream) {
  if (M != 1 || rate2 != 4 || N < 16 || N % 16 || K < 128 || K % 128 || S < 1 ||
      S > K / 64)
    return cudaErrorInvalidValue;
  dim3 grid((N + 63) / 64, S, 1);
  project<2, 0, true, true><<<grid, 128, 0, (cudaStream_t)stream>>>(
      (const unsigned*)packed, (const unsigned char*)lut, (const unsigned*)x,
      (const unsigned char*)xs, (float*)partial, 1, N, K, S);
  return cudaGetLastError();
}

__device__ __noinline__ float output_hadamard(float value) {
  __shared__ float exchange[128];
  for (int stride = 1; stride <= 16; stride *= 2) {
    float other = __shfl_xor_sync(0xffffffff, value, stride);
    value = (threadIdx.x & stride) ? other - value : value + other;
  }
  for (int stride = 32; stride <= 64; stride *= 2) {
    exchange[threadIdx.x] = value;
    __syncthreads();
    float other = exchange[threadIdx.x ^ stride];
    __syncthreads();
    value = (threadIdx.x & stride) ? other - value : value + other;
  }
  return value;
}

__global__ void fused_gemv(const half* input, const half* su, const half* sv,
                           const unsigned* packed, const unsigned char* table,
                           unsigned* q, unsigned char* qs, float* partial,
                           float* output, int N, int K, int S) {
  auto grid = cooperative_groups::this_grid();
  __shared__ unsigned char lut[1024];
  reinterpret_cast<unsigned long long*>(lut)[threadIdx.x] =
      reinterpret_cast<const unsigned long long*>(table)[threadIdx.x];
  for (int b = blockIdx.x; b < K / 128; b += gridDim.x) {
    pack_body<true, true>(input, q, qs, K, 1, su, b);
  }
  grid.sync();
  int tiles = (N + 63) / 64;
  for (int task = blockIdx.x; task < tiles * S; task += gridDim.x) {
    project_body<2, 0, true, true>(packed, lut, q, qs, partial, 1, N, K, S,
                                   task % tiles, task / tiles, 0);
  }
  grid.sync();
  for (int b = blockIdx.x; b < N / 128; b += gridDim.x) {
    int n = b * 128 + threadIdx.x;
    float value = 0;
    for (int j = 0; j < S; ++j)
      value += reinterpret_cast<volatile float*>(partial)[j * N + n];
    value = output_hadamard(value);
    output[n] = (value * 0.08838834764831845f) * __half2float(sv[n]);
  }
}

extern "C" int exl3_fp4_fused(const void* input, const void* su, const void* sv,
                              const void* packed, const void* lut, void* q,
                              void* qs, void* partial, void* output, int N,
                              int K, int S, int blocks, void* stream) {
  if (N < 128 || N % 128 || K < 128 || K % 128 || S < 1 || S > K / 64)
    return cudaErrorInvalidValue;
  int device, sms, active;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fused_gemv,
                                                         128, 0);
  if (status != cudaSuccess) return status;
  if (blocks <= 0) blocks = sms * active;
  if (blocks > sms * active) return cudaErrorInvalidValue;
  void* args[] = {&input, &su,      &sv,     &packed, &lut, &q,
                  &qs,    &partial, &output, &N,      &K,   &S};
  return cudaLaunchCooperativeKernel((void*)fused_gemv, dim3(blocks), dim3(128),
                                     args, 0, (cudaStream_t)stream);
}

__global__ void cta_fused_gemv(const half* input, const half* su,
                               const half* sv, const unsigned* packed,
                               const unsigned char* table, float* partial,
                               float* output, unsigned* counters, int N, int K,
                               int S) {
  __shared__ unsigned q[4 * 6144 / 8];
  __shared__ unsigned char qs[4 * 6144 / 16];
  __shared__ unsigned char lut[1024];
  __shared__ int last;
  reinterpret_cast<unsigned long long*>(lut)[threadIdx.x] =
      reinterpret_cast<const unsigned long long*>(table)[threadIdx.x];
  int G = K / 64;
  int start = G * blockIdx.y / S, stop = G * (blockIdx.y + 1) / S;
  for (int b = start / 2; b < (stop + 1) / 2; ++b)
    pack_body<true, true>(input, q, qs, K, 1, su, b);
  __syncthreads();
  project_body<2, 0, true, true>(packed, lut, q, qs, partial, 1, N, K, S,
                                 blockIdx.x * 2, blockIdx.y, 0);
  project_body<2, 0, true, true>(packed, lut, q, qs, partial, 1, N, K, S,
                                 blockIdx.x * 2 + 1, blockIdx.y, 0);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicInc(counters + blockIdx.x, S - 1) == S - 1;
  __syncthreads();
  if (last) {
    int n = blockIdx.x * 128 + threadIdx.x;
    float value = 0;
    for (int s = 0; s < S; ++s)
      value += reinterpret_cast<volatile float*>(partial)[s * N + n];
    value = output_hadamard(value);
    output[n] = value * 0.08838834764831845f * __half2float(sv[n]);
  }
}

extern "C" int exl3_fp4_cta_fused(const void* input, const void* su,
                                  const void* sv, const void* packed,
                                  const void* lut, void* partial, void* output,
                                  void* counters, int N, int K, int S,
                                  void* stream) {
  if (N < 128 || N % 128 || K < 128 || K % 128 || K > 6144 || S < 1 ||
      S > K / 64)
    return cudaErrorInvalidValue;
  cta_fused_gemv<<<dim3(N / 128, S), 128, 0, (cudaStream_t)stream>>>(
      (const half*)input, (const half*)su, (const half*)sv,
      (const unsigned*)packed, (const unsigned char*)lut, (float*)partial,
      (float*)output, (unsigned*)counters, N, K, S);
  return cudaGetLastError();
}
