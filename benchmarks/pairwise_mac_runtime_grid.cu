#include <cuda_runtime.h>

#include <cmath>
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
  temp1.hi += temp2.hi;
  temp1.hi *= prime;
  uint64_t res = in.lo - temp1.hi;
  if (res >= prime) res -= prime;
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
  for (uint64_t i = idx; i < n; i += stride) local += (a[i] != b[i]);
  if (local) atomicAdd(mismatches, local);
}

__global__ void mac_direct_runtime(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int nb,
    int nc,
    int L,
    int N) {
  constexpr int MAX_R = 32;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int limb = blockIdx.y;
  int lane = threadIdx.y;
  int by = blockDim.y;
  int R = (nb + by - 1) / by;
  if (x >= N) return;

  u128 sb[MAX_R];
  u128 sa[MAX_R];
  for (int r = 0; r < MAX_R; ++r) {
    sb[r] = {0, 0};
    sa[r] = {0, 0};
  }
  for (int i = 0; i < nc; ++i) {
    uint64_t bx = cipher[((0 * nc + i) * L + limb) * uint64_t(N) + x];
    uint64_t ax = cipher[((1 * nc + i) * L + limb) * uint64_t(N) + x];
    for (int r = 0; r < MAX_R; ++r) {
      int b = lane * R + r;
      if (r < R && b < nb) {
        uint64_t pv = plain[((b * nc + i) * L + limb) * uint64_t(N) + x];
        add_128(mul_64_64_128(bx, pv), sb[r]);
        add_128(mul_64_64_128(ax, pv), sa[r]);
      }
    }
  }
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];
  for (int r = 0; r < MAX_R; ++r) {
    int b = lane * R + r;
    if (r < R && b < nb) {
      out[((0 * nb + b) * L + limb) * uint64_t(N) + x] =
          barrett_reduce_128_64(sb[r], mod, ratio, k);
      out[((1 * nb + b) * L + limb) * uint64_t(N) + x] =
          barrett_reduce_128_64(sa[r], mod, ratio, k);
    }
  }
}

__global__ void mac_shared_runtime(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int nb,
    int nc,
    int L,
    int N) {
  constexpr int MAX_R = 32;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int limb = blockIdx.y;
  int lane = threadIdx.y;
  int by = blockDim.y;
  int R = (nb + by - 1) / by;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int block_threads = blockDim.x * blockDim.y;
  extern __shared__ uint64_t sh[];
  uint64_t* sh_bx = sh;
  uint64_t* sh_ax = sh + nc * blockDim.x;
  for (int idx = tid; idx < nc * blockDim.x; idx += block_threads) {
    int i = idx / blockDim.x;
    int local_x = idx - i * blockDim.x;
    int global_x = blockIdx.x * blockDim.x + local_x;
    uint64_t bx = 0, ax = 0;
    if (global_x < N) {
      bx = cipher[((0 * nc + i) * L + limb) * uint64_t(N) + global_x];
      ax = cipher[((1 * nc + i) * L + limb) * uint64_t(N) + global_x];
    }
    sh_bx[idx] = bx;
    sh_ax[idx] = ax;
  }
  __syncthreads();
  if (x >= N) return;

  u128 sb[MAX_R];
  u128 sa[MAX_R];
  for (int r = 0; r < MAX_R; ++r) {
    sb[r] = {0, 0};
    sa[r] = {0, 0};
  }
  for (int i = 0; i < nc; ++i) {
    uint64_t bx = sh_bx[i * blockDim.x + threadIdx.x];
    uint64_t ax = sh_ax[i * blockDim.x + threadIdx.x];
    for (int r = 0; r < MAX_R; ++r) {
      int b = lane * R + r;
      if (r < R && b < nb) {
        uint64_t pv = plain[((b * nc + i) * L + limb) * uint64_t(N) + x];
        add_128(mul_64_64_128(bx, pv), sb[r]);
        add_128(mul_64_64_128(ax, pv), sa[r]);
      }
    }
  }
  uint64_t mod = mods[limb];
  uint64_t ratio = ratios[limb];
  uint64_t k = ks[limb];
  for (int r = 0; r < MAX_R; ++r) {
    int b = lane * R + r;
    if (r < R && b < nb) {
      out[((0 * nb + b) * L + limb) * uint64_t(N) + x] =
          barrett_reduce_128_64(sb[r], mod, ratio, k);
      out[((1 * nb + b) * L + limb) * uint64_t(N) + x] =
          barrett_reduce_128_64(sa[r], mod, ratio, k);
    }
  }
}

__global__ void mac_split1_runtime(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ cipher,
    const uint64_t* __restrict__ plain,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks,
    int nb,
    int nc,
    int L,
    int N,
    int b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int limb = blockIdx.y;
  if (x >= N) return;
  u128 sb{0, 0}, sa{0, 0};
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
  out[((0 * nb + b) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sb, mod, ratio, k);
  out[((1 * nb + b) * L + limb) * uint64_t(N) + x] =
      barrett_reduce_128_64(sa, mod, ratio, k);
}

using KernelFn = void (*)(
    uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*,
    const uint64_t*, const uint64_t*, int, int, int, int);

uint64_t ratio_for_prime(uint64_t prime, uint64_t k) {
  return uint64_t(((__uint128_t)1 << k) / prime);
}

float run_kernel(
    KernelFn fn,
    dim3 grid,
    dim3 block,
    size_t smem,
    uint64_t* out,
    const uint64_t* cipher,
    const uint64_t* plain,
    const uint64_t* mods,
    const uint64_t* ratios,
    const uint64_t* ks,
    int nb,
    int nc,
    int L,
    int N,
    int warmup,
    int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn<<<grid, block, smem>>>(out, cipher, plain, mods, ratios, ks, nb, nc, L, N);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn<<<grid, block, smem>>>(out, cipher, plain, mods, ratios, ks, nb, nc, L, N);
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

float run_split1(
    dim3 grid,
    dim3 block,
    uint64_t* out,
    const uint64_t* cipher,
    const uint64_t* plain,
    const uint64_t* mods,
    const uint64_t* ratios,
    const uint64_t* ks,
    int nb,
    int nc,
    int L,
    int N,
    int warmup,
    int iters) {
  for (int w = 0; w < warmup; ++w) {
    for (int b = 0; b < nb; ++b) {
      mac_split1_runtime<<<grid, block>>>(out, cipher, plain, mods, ratios, ks, nb, nc, L, N, b);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    for (int b = 0; b < nb; ++b) {
      mac_split1_runtime<<<grid, block>>>(out, cipher, plain, mods, ratios, ks, nb, nc, L, N, b);
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

struct Args {
  int N = 4096;
  int L = 6;
  int iters = 5;
  int warmup = 2;
  int device = 0;
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
  }
  return args;
}

void run_case(int nb, int nc, const Args& args) {
  const uint64_t prime = 4503599627370449ULL;
  const uint64_t k = 112;
  const uint64_t ratio = ratio_for_prime(prime, k);
  size_t cipher_elems = size_t(2) * nc * args.L * args.N;
  size_t plain_elems = size_t(nb) * nc * args.L * args.N;
  size_t out_elems = size_t(2) * nb * args.L * args.N;
  uint64_t *cipher = nullptr, *plain = nullptr, *dir = nullptr, *out = nullptr;
  uint64_t *mods = nullptr, *ratios = nullptr, *ks = nullptr;
  unsigned long long* mismatch = nullptr;
  CUDA_CHECK(cudaMalloc(&cipher, cipher_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&plain, plain_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&dir, out_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&mods, args.L * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ratios, args.L * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ks, args.L * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&mismatch, sizeof(unsigned long long)));
  std::vector<uint64_t> hmods(args.L, prime), hratios(args.L, ratio), hks(args.L, k);
  CUDA_CHECK(cudaMemcpy(mods, hmods.data(), args.L * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ratios, hratios.data(), args.L * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ks, hks.data(), args.L * sizeof(uint64_t), cudaMemcpyHostToDevice));
  init_inputs_kernel<<<4096, 256>>>(cipher, plain, cipher_elems, plain_elems, prime);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 block_dir(32, 8);
  dim3 grid_dir((args.N + 31) / 32, args.L);
  float dir_us = run_kernel(mac_direct_runtime, grid_dir, block_dir, 0, dir, cipher, plain, mods, ratios, ks, nb, nc, args.L, args.N, args.warmup, args.iters);
  std::printf("NB=%-3d NC=%-3d DIR=%9.3fus", nb, nc, dir_us);

  size_t shared_bytes = size_t(2) * nc * 32 * sizeof(uint64_t);
  if (shared_bytes <= 160 * 1024) {
    CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(uint64_t)));
    float shr_us = run_kernel(mac_shared_runtime, grid_dir, block_dir, shared_bytes, out, cipher, plain, mods, ratios, ks, nb, nc, args.L, args.N, args.warmup, args.iters);
    CUDA_CHECK(cudaMemset(mismatch, 0, sizeof(unsigned long long)));
    compare_kernel<<<4096, 256>>>(dir, out, out_elems, mismatch);
    unsigned long long mm = 0;
    CUDA_CHECK(cudaMemcpy(&mm, mismatch, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::printf(" SHR=%9.3fus SHR/DIR=%6.3f mm=%llu", shr_us, double(shr_us) / dir_us, mm);
  } else {
    std::printf(" SHR=NA");
  }

  CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(uint64_t)));
  dim3 block_split(256);
  dim3 grid_split((args.N + 255) / 256, args.L);
  float split_us = run_split1(grid_split, block_split, out, cipher, plain, mods, ratios, ks, nb, nc, args.L, args.N, args.warmup, args.iters);
  CUDA_CHECK(cudaMemset(mismatch, 0, sizeof(unsigned long long)));
  compare_kernel<<<4096, 256>>>(dir, out, out_elems, mismatch);
  unsigned long long mm = 0;
  CUDA_CHECK(cudaMemcpy(&mm, mismatch, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::printf(" SPLIT1=%9.3fus SPLIT1/DIR=%6.3f mm=%llu\n", split_us, double(split_us) / dir_us, mm);

  CUDA_CHECK(cudaFree(cipher));
  CUDA_CHECK(cudaFree(plain));
  CUDA_CHECK(cudaFree(dir));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(mods));
  CUDA_CHECK(cudaFree(ratios));
  CUDA_CHECK(cudaFree(ks));
  CUDA_CHECK(cudaFree(mismatch));
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(args.device));
  CUDA_CHECK(cudaFuncSetAttribute(
      mac_shared_runtime,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));
  CUDA_CHECK(cudaFuncSetAttribute(
      mac_shared_runtime,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      160 * 1024));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
  std::printf("# runtime extended grid device=%d %s sm_%d%d N=%d L=%d iters=%d\n", args.device, prop.name, prop.major, prop.minor, args.N, args.L, args.iters);
  int values[] = {2, 4, 8, 9, 16, 32, 64, 128, 256};
  for (int nb : values) {
    for (int nc : values) {
      run_case(nb, nc, args);
    }
  }
  return 0;
}
