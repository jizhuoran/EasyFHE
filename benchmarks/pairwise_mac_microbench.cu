#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct u128 {
  uint64_t hi;
  uint64_t lo;
};

__device__ __forceinline__ u128 mul_64_64_128(uint64_t a, uint64_t b) {
  return {__umul64hi(a, b), a * b};
}

__device__ __forceinline__ void add_128(u128 v, u128& acc) {
  asm volatile("add.cc.u64 %1, %3, %1;\n\t"
               "addc.u64 %0, %2, %0;"
               : "+l"(acc.hi), "+l"(acc.lo)
               : "l"(v.hi), "l"(v.lo));
}

__device__ __forceinline__ uint64_t barrett_reduce_128_64(
    u128 in, uint64_t prime, uint64_t ratio, uint64_t k) {
  u128 temp1 = mul_64_64_128(in.lo, ratio);
  u128 temp2 = mul_64_64_128(in.hi, ratio);
  asm volatile("add.cc.u64 %0, %0, %1;" : "+l"(temp1.hi) : "l"(temp2.lo));
  asm volatile("addc.u64 %0, %0, %1;" : "+l"(temp2.hi) : "l"(uint64_t(0)));
  temp1.hi >>= k - 64;
  temp2.hi <<= 128 - k;
  temp1.hi = temp1.hi + temp2.hi;
  temp1.hi = temp1.hi * prime;
  uint64_t res = in.lo - temp1.hi;
  if (res >= prime) {
    res -= prime;
  }
  return res;
}

__device__ __forceinline__ uint64_t pseudo_value(uint64_t x, uint64_t prime) {
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  x *= 2685821657736338717ULL;
  return x % prime;
}

__global__ void init_inputs_kernel(
    uint64_t* cipher,
    uint64_t* plain,
    uint64_t cipher_elems,
    uint64_t plain_elems,
    uint64_t prime) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
  for (uint64_t i = idx; i < cipher_elems; i += stride) {
    cipher[i] = pseudo_value(i + 0x1234ULL, prime);
  }
  for (uint64_t i = idx; i < plain_elems; i += stride) {
    plain[i] = pseudo_value(i + 0xabcdefULL, prime);
  }
}

__global__ void compare_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t n,
    unsigned long long* mismatches) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
  unsigned long long local = 0;
  for (uint64_t i = idx; i < n; i += stride) {
    local += (a[i] != b[i]);
  }
  if (local) {
    atomicAdd(mismatches, local);
  }
}

template <int NB, int NC, int X>
__global__ void mac_original_shared(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int L,
    int N) {
  int x = blockIdx.x * X + threadIdx.x;
  int limb = blockIdx.y;
  if (x >= N) {
    return;
  }
  extern __shared__ uint64_t sh[];
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];

  u128 sum_bx{0, 0};
  u128 sum_ax{0, 0};
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    uint64_t pv = plain[(i * L + limb) * uint64_t(N) + x];
    uint64_t bx = cipher[((0 * NC + i) * L + limb) * uint64_t(N) + x];
    uint64_t ax = cipher[((1 * NC + i) * L + limb) * uint64_t(N) + x];
    add_128(mul_64_64_128(bx, pv), sum_bx);
    add_128(mul_64_64_128(ax, pv), sum_ax);
    sh[i * X + threadIdx.x] = bx;
    sh[(NC + i) * X + threadIdx.x] = ax;
  }
  out[((0 * NB + 0) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sum_bx, mod, ratio, k);
  out[((1 * NB + 0) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sum_ax, mod, ratio, k);

  __syncthreads();
#pragma unroll
  for (int b = 1; b < NB; ++b) {
    u128 sb{0, 0};
    u128 sa{0, 0};
#pragma unroll
    for (int i = 0; i < NC; ++i) {
      uint64_t pv = plain[((b * NC + i) * L + limb) * uint64_t(N) + x];
      uint64_t bx = sh[i * X + threadIdx.x];
      uint64_t ax = sh[(NC + i) * X + threadIdx.x];
      add_128(mul_64_64_128(bx, pv), sb);
      add_128(mul_64_64_128(ax, pv), sa);
    }
    out[((0 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sb, mod, ratio, k);
    out[((1 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sa, mod, ratio, k);
  }
}

template <int NB, int NC, int X, int BY>
__global__ void mac_regvec_cipher_2d(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int L,
    int N) {
  int x = blockIdx.x * X + threadIdx.x;
  int limb = blockIdx.y;
  int lane = threadIdx.y;
  if (x >= N) {
    return;
  }
  uint64_t cbx[NC];
  uint64_t cax[NC];
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    cbx[i] = cipher[((0 * NC + i) * L + limb) * uint64_t(N) + x];
    cax[i] = cipher[((1 * NC + i) * L + limb) * uint64_t(N) + x];
  }
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];
#pragma unroll
  for (int b = lane; b < NB; b += BY) {
    u128 sb{0, 0};
    u128 sa{0, 0};
#pragma unroll
    for (int i = 0; i < NC; ++i) {
      uint64_t pv = plain[((b * NC + i) * L + limb) * uint64_t(N) + x];
      add_128(mul_64_64_128(cbx[i], pv), sb);
      add_128(mul_64_64_128(cax[i], pv), sa);
    }
    out[((0 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sb, mod, ratio, k);
    out[((1 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sa, mod, ratio, k);
  }
}

template <int NB, int NC, int X, int BY, int R>
__global__ void mac_direct_batchtile_2d(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int L,
    int N) {
  static_assert(BY * R == NB, "hot path expects BY * R == NB");
  int x = blockIdx.x * X + threadIdx.x;
  int limb = blockIdx.y;
  int lane = threadIdx.y;
  if (x >= N) {
    return;
  }
  u128 sb[R];
  u128 sa[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    sb[r] = {0, 0};
    sa[r] = {0, 0};
  }
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    uint64_t bx = cipher[((0 * NC + i) * L + limb) * uint64_t(N) + x];
    uint64_t ax = cipher[((1 * NC + i) * L + limb) * uint64_t(N) + x];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      int b = lane * R + r;
      uint64_t pv = plain[((b * NC + i) * L + limb) * uint64_t(N) + x];
      add_128(mul_64_64_128(bx, pv), sb[r]);
      add_128(mul_64_64_128(ax, pv), sa[r]);
    }
  }
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    int b = lane * R + r;
    out[((0 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sb[r], mod, ratio, k);
    out[((1 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sa[r], mod, ratio, k);
  }
}

template <int NB, int NC, int X, int BY, int R>
__global__ void mac_shared_cipher_batchtile_2d(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int L,
    int N) {
  static_assert(BY * R == NB, "hot path expects BY * R == NB");
  int x = blockIdx.x * X + threadIdx.x;
  int limb = blockIdx.y;
  int lane = threadIdx.y;
  int tid = threadIdx.y * X + threadIdx.x;
  int block_threads = X * BY;

  extern __shared__ uint64_t sh[];
  uint64_t* sh_bx = sh;
  uint64_t* sh_ax = sh + NC * X;
  for (int idx = tid; idx < NC * X; idx += block_threads) {
    int i = idx / X;
    int local_x = idx - i * X;
    int global_x = blockIdx.x * X + local_x;
    uint64_t bx = 0;
    uint64_t ax = 0;
    if (global_x < N) {
      bx = cipher[((0 * NC + i) * L + limb) * uint64_t(N) + global_x];
      ax = cipher[((1 * NC + i) * L + limb) * uint64_t(N) + global_x];
    }
    sh_bx[idx] = bx;
    sh_ax[idx] = ax;
  }
  __syncthreads();
  if (x >= N) {
    return;
  }

  u128 sb[R];
  u128 sa[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    sb[r] = {0, 0};
    sa[r] = {0, 0};
  }
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    uint64_t bx = sh_bx[i * X + threadIdx.x];
    uint64_t ax = sh_ax[i * X + threadIdx.x];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      int b = lane * R + r;
      uint64_t pv = plain[((b * NC + i) * L + limb) * uint64_t(N) + x];
      add_128(mul_64_64_128(bx, pv), sb[r]);
      add_128(mul_64_64_128(ax, pv), sa[r]);
    }
  }
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    int b = lane * R + r;
    out[((0 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sb[r], mod, ratio, k);
    out[((1 * NB + b) * L + limb) * uint64_t(N) + x] =
        barrett_reduce_128_64(sa[r], mod, ratio, k);
  }
}

template <int TOTAL_NB, int NC, int X>
__global__ void mac_one_batch_call(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int L,
    int N,
    int b) {
  int x = blockIdx.x * X + threadIdx.x;
  int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  u128 sb{0, 0};
  u128 sa{0, 0};
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    uint64_t pv = plain[((b * NC + i) * L + limb) * uint64_t(N) + x];
    uint64_t bx = cipher[((0 * NC + i) * L + limb) * uint64_t(N) + x];
    uint64_t ax = cipher[((1 * NC + i) * L + limb) * uint64_t(N) + x];
    add_128(mul_64_64_128(bx, pv), sb);
    add_128(mul_64_64_128(ax, pv), sa);
  }
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];
  out[((0 * TOTAL_NB + b) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sb, mod, ratio, k);
  out[((1 * TOTAL_NB + b) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sa, mod, ratio, k);
}

__global__ void mac_one_batch_call_runtime(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int total_nb,
    int nc,
    int L,
    int N,
    int b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  u128 sb{0, 0};
  u128 sa{0, 0};
  for (int i = 0; i < nc; ++i) {
    uint64_t pv = plain[((b * nc + i) * L + limb) * uint64_t(N) + x];
    uint64_t bx = cipher[((0 * nc + i) * L + limb) * uint64_t(N) + x];
    uint64_t ax = cipher[((1 * nc + i) * L + limb) * uint64_t(N) + x];
    add_128(mul_64_64_128(bx, pv), sb);
    add_128(mul_64_64_128(ax, pv), sa);
  }
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];
  out[((0 * total_nb + b) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sb, mod, ratio, k);
  out[((1 * total_nb + b) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sa, mod, ratio, k);
}

using KernelFn = void (*)(
    uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*,
    const uint64_t*, const uint64_t*, int, int);

struct Variant {
  const char* id;
  const char* family;
  int NB;
  int NC;
  int X;
  int BY;
  int R;
  size_t shared_bytes;
  KernelFn fn;
};

template <typename Kernel>
float run_timed(
    Kernel kernel,
    dim3 grid,
    dim3 block,
    size_t shared_bytes,
    uint64_t* out,
    const uint64_t* cipher,
    const uint64_t* plain,
    const uint64_t* mods,
    const uint64_t* ratios,
    const uint64_t* ks,
    int L,
    int N,
    int warmup,
    int iters) {
  for (int i = 0; i < warmup; ++i) {
    kernel<<<grid, block, shared_bytes>>>(
        out, cipher, plain, mods, ratios, ks, L, N);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    kernel<<<grid, block, shared_bytes>>>(
        out, cipher, plain, mods, ratios, ks, L, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 1000.0f * ms / float(iters);
}

template <int TOTAL_NB, int NC, int X>
float run_one_batch_calls_timed(
    dim3 grid,
    dim3 block,
    uint64_t* out,
    const uint64_t* cipher,
    const uint64_t* plain,
    const uint64_t* mods,
    const uint64_t* ratios,
    const uint64_t* ks,
    int L,
    int N,
    int warmup,
    int iters) {
  for (int w = 0; w < warmup; ++w) {
    for (int b = 0; b < TOTAL_NB; ++b) {
      mac_one_batch_call<TOTAL_NB, NC, X><<<grid, block>>>(
          out, cipher, plain, mods, ratios, ks, L, N, b);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    for (int b = 0; b < TOTAL_NB; ++b) {
      mac_one_batch_call<TOTAL_NB, NC, X><<<grid, block>>>(
          out, cipher, plain, mods, ratios, ks, L, N, b);
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 1000.0f * ms / float(iters);
}

float run_one_batch_calls_runtime_timed(
    dim3 grid,
    dim3 block,
    uint64_t* out,
    const uint64_t* cipher,
    const uint64_t* plain,
    const uint64_t* mods,
    const uint64_t* ratios,
    const uint64_t* ks,
    int total_nb,
    int nc,
    int L,
    int N,
    int warmup,
    int iters) {
  for (int w = 0; w < warmup; ++w) {
    for (int b = 0; b < total_nb; ++b) {
      mac_one_batch_call_runtime<<<grid, block>>>(
          out, cipher, plain, mods, ratios, ks, total_nb, nc, L, N, b);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    for (int b = 0; b < total_nb; ++b) {
      mac_one_batch_call_runtime<<<grid, block>>>(
          out, cipher, plain, mods, ratios, ks, total_nb, nc, L, N, b);
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 1000.0f * ms / float(iters);
}

template <int NB, int NC, int X>
void configure_original(size_t shared_bytes) {
  auto fn = mac_original_shared<NB, NC, X>;
  CUDA_CHECK(cudaFuncSetAttribute(
      fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  if (shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes)));
  }
}

template <int NB, int NC, int X, int BY, int R>
void configure_shared(size_t shared_bytes) {
  auto fn = mac_shared_cipher_batchtile_2d<NB, NC, X, BY, R>;
  CUDA_CHECK(cudaFuncSetAttribute(
      fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  if (shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes)));
  }
}

uint64_t ratio_for_prime(uint64_t prime, uint64_t k) {
  return uint64_t(((__uint128_t)1 << k) / prime);
}

struct Args {
  int N = 65536;
  int L = 12;
  int iters = 50;
  int warmup = 10;
  int device = 0;
  std::string suite = "all";
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto take = [&](const char* name) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--N")) args.N = std::atoi(take("--N"));
    else if (!std::strcmp(argv[i], "--L")) args.L = std::atoi(take("--L"));
    else if (!std::strcmp(argv[i], "--iters")) args.iters = std::atoi(take("--iters"));
    else if (!std::strcmp(argv[i], "--warmup")) args.warmup = std::atoi(take("--warmup"));
    else if (!std::strcmp(argv[i], "--device")) args.device = std::atoi(take("--device"));
    else if (!std::strcmp(argv[i], "--suite")) args.suite = take("--suite");
    else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      std::exit(2);
    }
  }
  return args;
}

void configure_variant(const Variant& v) {
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(v.fn),
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));
  if (v.shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(v.fn),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(v.shared_bytes)));
  }
}

void run_suite(int NB, int NC, const std::vector<Variant>& variants, const Args& args) {
  std::printf("\n# suite NB=%d NC=%d N=%d L=%d iters=%d\n", NB, NC, args.N, args.L, args.iters);
  if (variants.empty()) {
    std::printf("# skipped: no variants\n");
    return;
  }

  const uint64_t prime = 4503599627370449ULL; // 52-bit-ish, fits k=112 ratio in uint64.
  const uint64_t k = 112;
  const uint64_t ratio = ratio_for_prime(prime, k);

  size_t cipher_elems = size_t(2) * NC * args.L * args.N;
  size_t plain_elems = size_t(NB) * NC * args.L * args.N;
  size_t out_elems = size_t(2) * NB * args.L * args.N;
  uint64_t *cipher = nullptr, *plain = nullptr, *baseline = nullptr, *out = nullptr;
  uint64_t *mods = nullptr, *ratios = nullptr, *ks = nullptr;
  unsigned long long* mismatch = nullptr;
  CUDA_CHECK(cudaMalloc(&cipher, cipher_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&plain, plain_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&baseline, out_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&mods, args.L * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ratios, args.L * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ks, args.L * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&mismatch, sizeof(unsigned long long)));

  std::vector<uint64_t> hmods(args.L, prime), hratios(args.L, ratio), hks(args.L, k);
  CUDA_CHECK(cudaMemcpy(mods, hmods.data(), args.L * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ratios, hratios.data(), args.L * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ks, hks.data(), args.L * sizeof(uint64_t), cudaMemcpyHostToDevice));
  int init_blocks = 4096;
  init_inputs_kernel<<<init_blocks, 256>>>(cipher, plain, cipher_elems, plain_elems, prime);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float baseline_us = -1.0f;
  const char* baseline_id = variants.front().id;
  for (size_t variant_idx = 0; variant_idx < variants.size(); ++variant_idx) {
    const auto& v = variants[variant_idx];
    configure_variant(v);
    dim3 block(v.X, v.BY);
    dim3 grid((args.N + v.X - 1) / v.X, args.L);
    CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(uint64_t)));
    float us = run_timed(
        v.fn, grid, block, v.shared_bytes, out, cipher, plain, mods, ratios, ks,
        args.L, args.N, args.warmup, args.iters);
    unsigned long long mismatches = 0;
    if (variant_idx == 0) {
      baseline_us = us;
      CUDA_CHECK(cudaMemcpy(baseline, out, out_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    } else {
      CUDA_CHECK(cudaMemset(mismatch, 0, sizeof(unsigned long long)));
      compare_kernel<<<4096, 256>>>(baseline, out, out_elems, mismatch);
      CUDA_CHECK(cudaMemcpy(&mismatches, mismatch, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }
    double rel = baseline_us > 0.0f ? double(us) / double(baseline_us) : 1.0;
    std::printf(
        "%-8s %-12s NB=%-2d NC=%-2d X=%-3d BY=%-2d R=%-2d smem=%6zuB time=%9.3fus rel_to_%s=%6.3f mismatches=%llu\n",
        v.id, v.family, v.NB, v.NC, v.X, v.BY, v.R, v.shared_bytes, us, baseline_id, rel, mismatches);
  }

  if (NB > 1) {
    dim3 block(256);
    dim3 grid((args.N + 255) / 256, args.L);
    CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(uint64_t)));
    float us = run_one_batch_calls_runtime_timed(
        grid, block, out, cipher, plain, mods, ratios, ks,
        NB, NC, args.L, args.N, args.warmup, args.iters);
    CUDA_CHECK(cudaMemset(mismatch, 0, sizeof(unsigned long long)));
    compare_kernel<<<4096, 256>>>(baseline, out, out_elems, mismatch);
    unsigned long long mismatches = 0;
    CUDA_CHECK(cudaMemcpy(&mismatches, mismatch, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    double rel = baseline_us > 0.0f ? double(us) / double(baseline_us) : 1.0;
    std::printf(
        "%-8s %-12s NB=%-2d NC=%-2d X=%-3d BY=%-2d R=%-2d smem=%6zuB time=%9.3fus rel_to_%s=%6.3f mismatches=%llu\n",
        "SPLIT1", "NB_CALLS", NB, NC, 256, 1, 1, size_t(0), us,
        baseline_id, rel, mismatches);
  }

  CUDA_CHECK(cudaFree(cipher));
  CUDA_CHECK(cudaFree(plain));
  CUDA_CHECK(cudaFree(baseline));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(mods));
  CUDA_CHECK(cudaFree(ratios));
  CUDA_CHECK(cudaFree(ks));
  CUDA_CHECK(cudaFree(mismatch));
}

template <int NB, int NC>
void run_grid_case(const Args& args) {
  constexpr int REG_BY = (NB >= 8) ? 8 : NB;
  constexpr int REG_X = 256 / REG_BY;
  constexpr int TILE_BY = (NB >= 8) ? 8 : NB;
  constexpr int TILE_R = NB / TILE_BY;
  constexpr int TILE_X = 256 / TILE_BY;
  constexpr size_t ORIGINAL_SMEM = 2ull * NC * 256 * sizeof(uint64_t);
  constexpr size_t SHARED_SMEM = 2ull * NC * TILE_X * sizeof(uint64_t);
  constexpr size_t MAX_DYN_SMEM = 160ull * 1024;

  std::vector<Variant> variants;
  if constexpr (ORIGINAL_SMEM <= MAX_DYN_SMEM) {
    variants.push_back({
        "ORIG", "ORIGINAL", NB, NC, 256, 1, NB, ORIGINAL_SMEM,
        mac_original_shared<NB, NC, 256>});
  }
  if constexpr (NC <= 64) {
    variants.push_back({
        "REG", "REGVEC", NB, NC, REG_X, REG_BY, (NB + REG_BY - 1) / REG_BY, 0,
        mac_regvec_cipher_2d<NB, NC, REG_X, REG_BY>});
  }
  variants.push_back({
      "DIR", "DIRECT", NB, NC, TILE_X, TILE_BY, TILE_R, 0,
      mac_direct_batchtile_2d<NB, NC, TILE_X, TILE_BY, TILE_R>});
  if constexpr (SHARED_SMEM <= MAX_DYN_SMEM) {
    variants.push_back({
        "SHR", "SHARED", NB, NC, TILE_X, TILE_BY, TILE_R, SHARED_SMEM,
        mac_shared_cipher_batchtile_2d<NB, NC, TILE_X, TILE_BY, TILE_R>});
  }
  run_suite(NB, NC, variants, args);
}

template <int NB>
void run_grid_row(const Args& args) {
  run_grid_case<NB, 2>(args);
  run_grid_case<NB, 4>(args);
  run_grid_case<NB, 8>(args);
  run_grid_case<NB, 9>(args);
  run_grid_case<NB, 16>(args);
  run_grid_case<NB, 32>(args);
  run_grid_case<NB, 64>(args);
}

void run_grid(const Args& args) {
  std::printf("# grid NB,NC in {2,4,8,9,16,32,64}; representative REG/DIR/SHR configs\n");
  run_grid_row<2>(args);
  run_grid_row<4>(args);
  run_grid_row<8>(args);
  run_grid_row<16>(args);
  run_grid_row<32>(args);
  run_grid_row<64>(args);
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
  std::printf("# device=%d %s sm_%d%d\n", args.device, prop.name, prop.major, prop.minor);

  configure_original<64, 9, 256>(2 * 9 * 256 * sizeof(uint64_t));
  configure_original<4, 32, 256>(2 * 32 * 256 * sizeof(uint64_t));
  configure_original<8, 32, 256>(2 * 32 * 256 * sizeof(uint64_t));
  configure_shared<64, 9, 32, 8, 8>(2 * 9 * 32 * sizeof(uint64_t));
  configure_shared<4, 32, 64, 4, 1>(2 * 32 * 64 * sizeof(uint64_t));
  configure_shared<8, 32, 64, 4, 2>(2 * 32 * 64 * sizeof(uint64_t));
  configure_shared<8, 32, 32, 8, 1>(2 * 32 * 32 * sizeof(uint64_t));

  std::vector<Variant> A = {
      {"A0", "ORIGINAL", 64, 9, 256, 1, 64, 2 * 9 * 256 * sizeof(uint64_t), mac_original_shared<64, 9, 256>},
      {"A1", "REGVEC", 64, 9, 32, 8, 8, 0, mac_regvec_cipher_2d<64, 9, 32, 8>},
      {"A2", "REGVEC", 64, 9, 64, 4, 16, 0, mac_regvec_cipher_2d<64, 9, 64, 4>},
      {"A3", "DIRECT", 64, 9, 32, 8, 8, 0, mac_direct_batchtile_2d<64, 9, 32, 8, 8>},
      {"A4", "SHARED", 64, 9, 32, 8, 8, 2 * 9 * 32 * sizeof(uint64_t), mac_shared_cipher_batchtile_2d<64, 9, 32, 8, 8>},
  };
  std::vector<Variant> B = {
      {"B0", "ORIGINAL", 4, 32, 256, 1, 4, 2 * 32 * 256 * sizeof(uint64_t), mac_original_shared<4, 32, 256>},
      {"B1", "DIRECT", 4, 32, 256, 1, 4, 0, mac_direct_batchtile_2d<4, 32, 256, 1, 4>},
      {"B2", "DIRECT", 4, 32, 128, 2, 2, 0, mac_direct_batchtile_2d<4, 32, 128, 2, 2>},
      {"B3", "DIRECT", 4, 32, 64, 4, 1, 0, mac_direct_batchtile_2d<4, 32, 64, 4, 1>},
      {"B4", "SHARED", 4, 32, 64, 4, 1, 2 * 32 * 64 * sizeof(uint64_t), mac_shared_cipher_batchtile_2d<4, 32, 64, 4, 1>},
  };
  std::vector<Variant> C = {
      {"C0", "ORIGINAL", 8, 32, 256, 1, 8, 2 * 32 * 256 * sizeof(uint64_t), mac_original_shared<8, 32, 256>},
      {"C1", "DIRECT", 8, 32, 128, 2, 4, 0, mac_direct_batchtile_2d<8, 32, 128, 2, 4>},
      {"C2", "DIRECT", 8, 32, 64, 4, 2, 0, mac_direct_batchtile_2d<8, 32, 64, 4, 2>},
      {"C3", "DIRECT", 8, 32, 32, 8, 1, 0, mac_direct_batchtile_2d<8, 32, 32, 8, 1>},
      {"C4", "SHARED", 8, 32, 64, 4, 2, 2 * 32 * 64 * sizeof(uint64_t), mac_shared_cipher_batchtile_2d<8, 32, 64, 4, 2>},
      {"C5", "SHARED", 8, 32, 32, 8, 1, 2 * 32 * 32 * sizeof(uint64_t), mac_shared_cipher_batchtile_2d<8, 32, 32, 8, 1>},
  };

  if (args.suite == "grid") {
    run_grid(args);
    return 0;
  }
  if (args.suite == "all" || args.suite == "A") run_suite(64, 9, A, args);
  if (args.suite == "all" || args.suite == "B") run_suite(4, 32, B, args);
  if (args.suite == "all" || args.suite == "C") run_suite(8, 32, C, args);
  return 0;
}
