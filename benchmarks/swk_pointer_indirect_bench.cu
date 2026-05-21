#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err__));                               \
      std::exit(1);                                                          \
    }                                                                        \
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

__device__ __forceinline__ uint64_t pseudo_value(uint64_t x) {
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  return x * 2685821657736338717ULL;
}

__global__ void init_kernel(uint64_t* ptr, uint64_t n, uint64_t seed) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
  for (uint64_t i = tid; i < n; i += stride) {
    ptr[i] = pseudo_value(i + seed);
  }
}

__device__ __forceinline__ uint64_t cheap_reduce(u128 v) {
  return v.lo ^ (v.hi * 0x9e3779b97f4a7c15ULL);
}

template <int NUM_KEYS, int BETA>
__global__ void merged_keys_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    const uint64_t* __restrict__ keys,
    int length,
    int N) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t key_stride = uint64_t(2 * BETA) * ln;
  const uint64_t out_stride = uint64_t(2) * ln;
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);

#pragma unroll
  for (int key = 0; key < NUM_KEYS; ++key) {
    u128 acc_bx{0, 0};
    u128 acc_ax{0, 0};
    const uint64_t* key_base = keys + uint64_t(key) * key_stride;
#pragma unroll
    for (int beta = 0; beta < BETA; ++beta) {
      const uint64_t in_val = input[uint64_t(beta) * ln + limb_x];
      const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
      const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
      add_128(mul_64_64_128(in_val, bx), acc_bx);
      add_128(mul_64_64_128(in_val, ax), acc_ax);
    }
    uint64_t* out_base = out + uint64_t(key) * out_stride;
    out_base[limb_x] = cheap_reduce(acc_bx);
    out_base[ln + limb_x] = cheap_reduce(acc_ax);
  }
}

template <int NUM_KEYS, int BETA>
__global__ void indirect_keys_global_ptr_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    const uint64_t* const* __restrict__ key_ptrs,
    int length,
    int N) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t out_stride = uint64_t(2) * ln;
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);

#pragma unroll
  for (int key = 0; key < NUM_KEYS; ++key) {
    u128 acc_bx{0, 0};
    u128 acc_ax{0, 0};
    const uint64_t* key_base = key_ptrs[key];
#pragma unroll
    for (int beta = 0; beta < BETA; ++beta) {
      const uint64_t in_val = input[uint64_t(beta) * ln + limb_x];
      const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
      const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
      add_128(mul_64_64_128(in_val, bx), acc_bx);
      add_128(mul_64_64_128(in_val, ax), acc_ax);
    }
    uint64_t* out_base = out + uint64_t(key) * out_stride;
    out_base[limb_x] = cheap_reduce(acc_bx);
    out_base[ln + limb_x] = cheap_reduce(acc_ax);
  }
}

template <int NUM_KEYS, int BETA>
__global__ void indirect_keys_shared_ptr_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    const uint64_t* const* __restrict__ key_ptrs,
    int length,
    int N) {
  __shared__ const uint64_t* ptrs[NUM_KEYS];
  if (threadIdx.x < NUM_KEYS) {
    ptrs[threadIdx.x] = key_ptrs[threadIdx.x];
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t out_stride = uint64_t(2) * ln;
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);

#pragma unroll
  for (int key = 0; key < NUM_KEYS; ++key) {
    u128 acc_bx{0, 0};
    u128 acc_ax{0, 0};
    const uint64_t* key_base = ptrs[key];
#pragma unroll
    for (int beta = 0; beta < BETA; ++beta) {
      const uint64_t in_val = input[uint64_t(beta) * ln + limb_x];
      const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
      const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
      add_128(mul_64_64_128(in_val, bx), acc_bx);
      add_128(mul_64_64_128(in_val, ax), acc_ax);
    }
    uint64_t* out_base = out + uint64_t(key) * out_stride;
    out_base[limb_x] = cheap_reduce(acc_bx);
    out_base[ln + limb_x] = cheap_reduce(acc_ax);
  }
}

template <int KEY_ID, int BETA>
__device__ __forceinline__ void compute_param_key(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    const uint64_t* __restrict__ key_base,
    uint64_t ln,
    uint64_t out_stride,
    uint64_t limb_x) {
  u128 acc_bx{0, 0};
  u128 acc_ax{0, 0};
#pragma unroll
  for (int beta = 0; beta < BETA; ++beta) {
    const uint64_t in_val = input[uint64_t(beta) * ln + limb_x];
    const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
    const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
    add_128(mul_64_64_128(in_val, bx), acc_bx);
    add_128(mul_64_64_128(in_val, ax), acc_ax);
  }
  uint64_t* out_base = out + uint64_t(KEY_ID) * out_stride;
  out_base[limb_x] = cheap_reduce(acc_bx);
  out_base[ln + limb_x] = cheap_reduce(acc_ax);
}

template <int BETA, int... KEY_IDS, typename... KeyPtrs>
__device__ __forceinline__ void compute_param_keys(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    uint64_t ln,
    uint64_t out_stride,
    uint64_t limb_x,
    std::integer_sequence<int, KEY_IDS...>,
    KeyPtrs... key_ptrs) {
  (compute_param_key<KEY_IDS, BETA>(
       out, input, key_ptrs, ln, out_stride, limb_x),
   ...);
}

template <int BETA, typename... KeyPtrs>
__global__ void param_keys_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    int length,
    int N,
    KeyPtrs... key_ptrs) {
  constexpr int NUM_KEYS = sizeof...(KeyPtrs);
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t out_stride = uint64_t(2) * ln;
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);
  compute_param_keys<BETA>(
      out,
      input,
      ln,
      out_stride,
      limb_x,
      std::make_integer_sequence<int, NUM_KEYS>{},
      key_ptrs...);
}

template <int TILE, int BETA, typename... KeyPtrs>
__global__ void param_keys_cached_tile_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    int length,
    int N,
    KeyPtrs... key_ptrs) {
  constexpr int NUM_KEYS = sizeof...(KeyPtrs);
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t out_stride = uint64_t(2) * ln;
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);
  const uint64_t* keys[NUM_KEYS] = {key_ptrs...};

#pragma unroll
  for (int start = 0; start < NUM_KEYS; start += TILE) {
    u128 acc_bx[TILE];
    u128 acc_ax[TILE];
#pragma unroll
    for (int t = 0; t < TILE; ++t) {
      acc_bx[t] = {0, 0};
      acc_ax[t] = {0, 0};
    }

#pragma unroll
    for (int beta = 0; beta < BETA; ++beta) {
      const uint64_t in_val = input[uint64_t(beta) * ln + limb_x];
#pragma unroll
      for (int t = 0; t < TILE; ++t) {
        if (start + t < NUM_KEYS) {
          const uint64_t* key_base = keys[start + t];
          const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
          const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
          add_128(mul_64_64_128(in_val, bx), acc_bx[t]);
          add_128(mul_64_64_128(in_val, ax), acc_ax[t]);
        }
      }
    }

#pragma unroll
    for (int t = 0; t < TILE; ++t) {
      if (start + t < NUM_KEYS) {
        uint64_t* out_base = out + uint64_t(start + t) * out_stride;
        out_base[limb_x] = cheap_reduce(acc_bx[t]);
        out_base[ln + limb_x] = cheap_reduce(acc_ax[t]);
      }
    }
  }
}

template <int... KEY_IDS, typename... KeyPtrs>
__device__ __forceinline__ const uint64_t* select_key_ptr_impl(
    int key,
    std::integer_sequence<int, KEY_IDS...>,
    KeyPtrs... key_ptrs) {
  const uint64_t* selected = nullptr;
  ((selected = (key == KEY_IDS ? key_ptrs : selected)), ...);
  return selected;
}

template <typename... KeyPtrs>
__device__ __forceinline__ const uint64_t* select_key_ptr(
    int key,
    KeyPtrs... key_ptrs) {
  return select_key_ptr_impl(
      key, std::make_integer_sequence<int, sizeof...(KeyPtrs)>{}, key_ptrs...);
}

template <int BETA, typename... KeyPtrs>
__global__ void param_key_grid_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    int length,
    int N,
    KeyPtrs... key_ptrs) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  const int key = blockIdx.z;
  if (x >= N) {
    return;
  }

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t out_stride = uint64_t(2) * ln;
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);
  const uint64_t* key_base = select_key_ptr(key, key_ptrs...);

  u128 acc_bx{0, 0};
  u128 acc_ax{0, 0};
#pragma unroll
  for (int beta = 0; beta < BETA; ++beta) {
    const uint64_t in_val = input[uint64_t(beta) * ln + limb_x];
    const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
    const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
    add_128(mul_64_64_128(in_val, bx), acc_bx);
    add_128(mul_64_64_128(in_val, ax), acc_ax);
  }

  uint64_t* out_base = out + uint64_t(key) * out_stride;
  out_base[limb_x] = cheap_reduce(acc_bx);
  out_base[ln + limb_x] = cheap_reduce(acc_ax);
}

template <int KEY_TILE, int X, int BETA, typename... KeyPtrs>
__global__ void param_key_grid_shared_digits_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ input,
    int length,
    int N,
    KeyPtrs... key_ptrs) {
  constexpr int NUM_KEYS = sizeof...(KeyPtrs);
  __shared__ uint64_t digits[BETA][X];

  const int x = blockIdx.x * X + threadIdx.x;
  const int limb = blockIdx.y;
  const int key = blockIdx.z * KEY_TILE + threadIdx.y;

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);
  if (threadIdx.y == 0) {
#pragma unroll
    for (int beta = 0; beta < BETA; ++beta) {
      digits[beta][threadIdx.x] =
          (x < N) ? input[uint64_t(beta) * ln + limb_x] : 0;
    }
  }
  __syncthreads();

  if (x >= N || key >= NUM_KEYS) {
    return;
  }

  const uint64_t out_stride = uint64_t(2) * ln;
  const uint64_t* key_base = select_key_ptr(key, key_ptrs...);
  u128 acc_bx{0, 0};
  u128 acc_ax{0, 0};
#pragma unroll
  for (int beta = 0; beta < BETA; ++beta) {
    const uint64_t in_val = digits[beta][threadIdx.x];
    const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
    const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
    add_128(mul_64_64_128(in_val, bx), acc_bx);
    add_128(mul_64_64_128(in_val, ax), acc_ax);
  }

  uint64_t* out_base = out + uint64_t(key) * out_stride;
  out_base[limb_x] = cheap_reduce(acc_bx);
  out_base[ln + limb_x] = cheap_reduce(acc_ax);
}

template <int BETA>
__global__ void single_key_kernel(
    uint64_t* __restrict__ out_key,
    const uint64_t* __restrict__ input,
    const uint64_t* __restrict__ key_base,
    int length,
    int N) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  if (x >= N) {
    return;
  }

  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t limb_x = uint64_t(limb) * uint64_t(N) + uint64_t(x);
  u128 acc_bx{0, 0};
  u128 acc_ax{0, 0};
#pragma unroll
  for (int beta = 0; beta < BETA; ++beta) {
    const uint64_t in_val = input[uint64_t(beta) * ln + limb_x];
    const uint64_t bx = key_base[uint64_t(beta) * ln + limb_x];
    const uint64_t ax = key_base[uint64_t(BETA + beta) * ln + limb_x];
    add_128(mul_64_64_128(in_val, bx), acc_bx);
    add_128(mul_64_64_128(in_val, ax), acc_ax);
  }
  out_key[limb_x] = cheap_reduce(acc_bx);
  out_key[ln + limb_x] = cheap_reduce(acc_ax);
}

using BenchFn = void (*)(uint64_t*, const uint64_t*, const uint64_t*,
                        const uint64_t* const*, int, int, dim3, dim3);

template <int NUM_KEYS, int BETA>
void launch_merged(
    uint64_t* out,
    const uint64_t* input,
    const uint64_t* merged_keys,
    const uint64_t* const*,
    int length,
    int N,
    dim3 grid,
    dim3 block) {
  merged_keys_kernel<NUM_KEYS, BETA><<<grid, block>>>(
      out, input, merged_keys, length, N);
}

template <int NUM_KEYS, int BETA>
void launch_indirect_global(
    uint64_t* out,
    const uint64_t* input,
    const uint64_t*,
    const uint64_t* const* key_ptrs,
    int length,
    int N,
    dim3 grid,
    dim3 block) {
  indirect_keys_global_ptr_kernel<NUM_KEYS, BETA><<<grid, block>>>(
      out, input, key_ptrs, length, N);
}

template <int NUM_KEYS, int BETA>
void launch_indirect_shared(
    uint64_t* out,
    const uint64_t* input,
    const uint64_t*,
    const uint64_t* const* key_ptrs,
    int length,
    int N,
    dim3 grid,
    dim3 block) {
  indirect_keys_shared_ptr_kernel<NUM_KEYS, BETA><<<grid, block>>>(
      out, input, key_ptrs, length, N);
}

template <int NUM_KEYS, int BETA>
struct ParamLauncher;

template <int NUM_KEYS, int BETA, int TILE>
struct ParamCachedTileLauncher;

template <int NUM_KEYS, int BETA>
struct ParamKeyGridLauncher;

template <int NUM_KEYS, int BETA, int KEY_TILE, int X>
struct ParamSharedDigitsLauncher;

#define DEFINE_PARAM_LAUNCHER(NUM, ...)                                      \
  template <int BETA>                                                        \
  struct ParamLauncher<NUM, BETA> {                                          \
    static void launch(                                                      \
        uint64_t* out,                                                       \
        const uint64_t* input,                                               \
        const std::vector<uint64_t*>& keys,                                  \
        int length,                                                          \
        int N,                                                               \
        dim3 grid,                                                           \
        dim3 block) {                                                        \
      param_keys_kernel<BETA><<<grid, block>>>(                              \
          out, input, length, N, __VA_ARGS__);                               \
    }                                                                        \
  };                                                                         \
  template <int BETA, int TILE>                                              \
  struct ParamCachedTileLauncher<NUM, BETA, TILE> {                          \
    static void launch(                                                      \
        uint64_t* out,                                                       \
        const uint64_t* input,                                               \
        const std::vector<uint64_t*>& keys,                                  \
        int length,                                                          \
        int N,                                                               \
        dim3 grid,                                                           \
        dim3 block) {                                                        \
      param_keys_cached_tile_kernel<TILE, BETA><<<grid, block>>>(            \
          out, input, length, N, __VA_ARGS__);                               \
    }                                                                        \
  };                                                                         \
  template <int BETA>                                                        \
  struct ParamKeyGridLauncher<NUM, BETA> {                                   \
    static void launch(                                                      \
        uint64_t* out,                                                       \
        const uint64_t* input,                                               \
        const std::vector<uint64_t*>& keys,                                  \
        int length,                                                          \
        int N,                                                               \
        dim3 grid,                                                           \
        dim3 block) {                                                        \
      param_key_grid_kernel<BETA><<<grid, block>>>(                          \
          out, input, length, N, __VA_ARGS__);                               \
    }                                                                        \
  };                                                                         \
  template <int BETA, int KEY_TILE, int X>                                   \
  struct ParamSharedDigitsLauncher<NUM, BETA, KEY_TILE, X> {                 \
    static void launch(                                                      \
        uint64_t* out,                                                       \
        const uint64_t* input,                                               \
        const std::vector<uint64_t*>& keys,                                  \
        int length,                                                          \
        int N,                                                               \
        dim3 grid,                                                           \
        dim3 block) {                                                        \
      param_key_grid_shared_digits_kernel<KEY_TILE, X, BETA>                 \
          <<<grid, block>>>(out, input, length, N, __VA_ARGS__);             \
    }                                                                        \
  }

DEFINE_PARAM_LAUNCHER(1, keys[0]);
DEFINE_PARAM_LAUNCHER(2, keys[0], keys[1]);
DEFINE_PARAM_LAUNCHER(4, keys[0], keys[1], keys[2], keys[3]);
DEFINE_PARAM_LAUNCHER(
    8,
    keys[0],
    keys[1],
    keys[2],
    keys[3],
    keys[4],
    keys[5],
    keys[6],
    keys[7]);
DEFINE_PARAM_LAUNCHER(
    16,
    keys[0],
    keys[1],
    keys[2],
    keys[3],
    keys[4],
    keys[5],
    keys[6],
    keys[7],
    keys[8],
    keys[9],
    keys[10],
    keys[11],
    keys[12],
    keys[13],
    keys[14],
    keys[15]);
DEFINE_PARAM_LAUNCHER(
    32,
    keys[0],
    keys[1],
    keys[2],
    keys[3],
    keys[4],
    keys[5],
    keys[6],
    keys[7],
    keys[8],
    keys[9],
    keys[10],
    keys[11],
    keys[12],
    keys[13],
    keys[14],
    keys[15],
    keys[16],
    keys[17],
    keys[18],
    keys[19],
    keys[20],
    keys[21],
    keys[22],
    keys[23],
    keys[24],
    keys[25],
    keys[26],
    keys[27],
    keys[28],
    keys[29],
    keys[30],
    keys[31]);

#undef DEFINE_PARAM_LAUNCHER

float time_kernel(
    BenchFn fn,
    uint64_t* out,
    const uint64_t* input,
    const uint64_t* merged_keys,
    const uint64_t* const* key_ptrs,
    int length,
    int N,
    dim3 grid,
    dim3 block,
    int iters) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) {
    fn(out, input, merged_keys, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn(out, input, merged_keys, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.0f / float(iters);
}

template <int NUM_KEYS, int BETA>
float time_separate_launches(
    uint64_t* out,
    const uint64_t* input,
    const std::vector<uint64_t*>& key_ptrs,
    int length,
    int N,
    dim3 grid,
    dim3 block,
    int iters) {
  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t out_stride = uint64_t(2) * ln;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) {
    for (int key = 0; key < NUM_KEYS; ++key) {
      single_key_kernel<BETA><<<grid, block>>>(
          out + uint64_t(key) * out_stride,
          input,
          key_ptrs[key],
          length,
          N);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    for (int key = 0; key < NUM_KEYS; ++key) {
      single_key_kernel<BETA><<<grid, block>>>(
          out + uint64_t(key) * out_stride,
          input,
          key_ptrs[key],
          length,
          N);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.0f / float(iters);
}

template <int NUM_KEYS, int BETA>
float time_param_kernel(
    uint64_t* out,
    const uint64_t* input,
    const std::vector<uint64_t*>& key_ptrs,
    int length,
    int N,
    dim3 grid,
    dim3 block,
    int iters) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) {
    ParamLauncher<NUM_KEYS, BETA>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    ParamLauncher<NUM_KEYS, BETA>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.0f / float(iters);
}

template <int NUM_KEYS, int BETA, int TILE>
float time_param_cached_tile_kernel(
    uint64_t* out,
    const uint64_t* input,
    const std::vector<uint64_t*>& key_ptrs,
    int length,
    int N,
    dim3 grid,
    dim3 block,
    int iters) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) {
    ParamCachedTileLauncher<NUM_KEYS, BETA, TILE>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    ParamCachedTileLauncher<NUM_KEYS, BETA, TILE>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.0f / float(iters);
}

template <int NUM_KEYS, int BETA>
float time_param_key_grid_kernel(
    uint64_t* out,
    const uint64_t* input,
    const std::vector<uint64_t*>& key_ptrs,
    int length,
    int N,
    int iters) {
  constexpr int x_block = 256;
  const dim3 block(x_block);
  const dim3 grid((N + x_block - 1) / x_block, length, NUM_KEYS);
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) {
    ParamKeyGridLauncher<NUM_KEYS, BETA>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    ParamKeyGridLauncher<NUM_KEYS, BETA>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.0f / float(iters);
}

template <int NUM_KEYS, int BETA, int KEY_TILE, int X>
float time_param_shared_digits_kernel(
    uint64_t* out,
    const uint64_t* input,
    const std::vector<uint64_t*>& key_ptrs,
    int length,
    int N,
    int iters) {
  const dim3 block(X, KEY_TILE);
  const dim3 grid((N + X - 1) / X, length, (NUM_KEYS + KEY_TILE - 1) / KEY_TILE);
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) {
    ParamSharedDigitsLauncher<NUM_KEYS, BETA, KEY_TILE, X>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    ParamSharedDigitsLauncher<NUM_KEYS, BETA, KEY_TILE, X>::launch(
        out, input, key_ptrs, length, N, grid, block);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.0f / float(iters);
}

template <int NUM_KEYS, int BETA>
void run_case(int length, int N, int iters) {
  constexpr int block_size = 256;
  const uint64_t ln = uint64_t(length) * uint64_t(N);
  const uint64_t input_elems = uint64_t(BETA) * ln;
  const uint64_t key_elems = uint64_t(2 * BETA) * ln;
  const uint64_t merged_elems = uint64_t(NUM_KEYS) * key_elems;
  const uint64_t out_elems = uint64_t(NUM_KEYS) * uint64_t(2) * ln;

  uint64_t* input = nullptr;
  uint64_t* merged_keys = nullptr;
  uint64_t* out = nullptr;
  CUDA_CHECK(cudaMalloc(&input, input_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&merged_keys, merged_elems * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(uint64_t)));

  std::vector<uint64_t*> host_keys(NUM_KEYS);
  for (int k = 0; k < NUM_KEYS; ++k) {
    CUDA_CHECK(cudaMalloc(&host_keys[k], key_elems * sizeof(uint64_t)));
  }
  const uint64_t** device_key_ptrs = nullptr;
  CUDA_CHECK(cudaMalloc(&device_key_ptrs, NUM_KEYS * sizeof(uint64_t*)));
  CUDA_CHECK(cudaMemcpy(
      device_key_ptrs,
      host_keys.data(),
      NUM_KEYS * sizeof(uint64_t*),
      cudaMemcpyHostToDevice));

  init_kernel<<<256, 256>>>(input, input_elems, 0x1234);
  init_kernel<<<1024, 256>>>(merged_keys, merged_elems, 0x5678);
  for (int k = 0; k < NUM_KEYS; ++k) {
    init_kernel<<<1024, 256>>>(host_keys[k], key_elems, 0x5678 + uint64_t(k) * key_elems);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const dim3 block(block_size);
  const dim3 grid((N + block_size - 1) / block_size, length);

  const float merged_us = time_kernel(
      &launch_merged<NUM_KEYS, BETA>,
      out,
      input,
      merged_keys,
      device_key_ptrs,
      length,
      N,
      grid,
      block,
      iters);
  const float indirect_global_us = time_kernel(
      &launch_indirect_global<NUM_KEYS, BETA>,
      out,
      input,
      merged_keys,
      device_key_ptrs,
      length,
      N,
      grid,
      block,
      iters);
  const float indirect_shared_us = time_kernel(
      &launch_indirect_shared<NUM_KEYS, BETA>,
      out,
      input,
      merged_keys,
      device_key_ptrs,
      length,
      N,
      grid,
      block,
      iters);
  const float param_us = time_param_kernel<NUM_KEYS, BETA>(
      out,
      input,
      host_keys,
      length,
      N,
      grid,
      block,
      iters);
  const float keygrid_us = time_param_key_grid_kernel<NUM_KEYS, BETA>(
      out,
      input,
      host_keys,
      length,
      N,
      iters);
  const float shared4x64_us =
      time_param_shared_digits_kernel<NUM_KEYS, BETA, 4, 64>(
          out,
          input,
          host_keys,
          length,
          N,
          iters);
  const float shared4x128_us =
      time_param_shared_digits_kernel<NUM_KEYS, BETA, 4, 128>(
          out,
          input,
          host_keys,
          length,
          N,
          iters);
  const float shared8x64_us =
      time_param_shared_digits_kernel<NUM_KEYS, BETA, 8, 64>(
          out,
          input,
          host_keys,
          length,
          N,
          iters);
  const float shared8x128_us =
      time_param_shared_digits_kernel<NUM_KEYS, BETA, 8, 128>(
          out,
          input,
          host_keys,
          length,
          N,
          iters);
  float shared_best_us = shared4x64_us;
  const char* shared_best_name = "sh4x64";
  if (shared4x128_us < shared_best_us) {
    shared_best_us = shared4x128_us;
    shared_best_name = "sh4x128";
  }
  if (shared8x64_us < shared_best_us) {
    shared_best_us = shared8x64_us;
    shared_best_name = "sh8x64";
  }
  if (shared8x128_us < shared_best_us) {
    shared_best_us = shared8x128_us;
    shared_best_name = "sh8x128";
  }
  const float separate_us = time_separate_launches<NUM_KEYS, BETA>(
      out,
      input,
      host_keys,
      length,
      N,
      grid,
      block,
      iters);

  std::printf(
      "keys=%2d beta=%d length=%2d  separate=%8.1f us  serial=%8.1f us (%+5.1f%%)  keygrid=%8.1f us (%+5.1f%%)  shared=%8.1f us %-7s (%+5.1f%%)  merged=%8.1f us (%+5.1f%%)  ptr=%8.1f us (%+5.1f%%)\n",
      NUM_KEYS,
      BETA,
      length,
      separate_us,
      param_us,
      (param_us / separate_us - 1.0f) * 100.0f,
      keygrid_us,
      (keygrid_us / separate_us - 1.0f) * 100.0f,
      shared_best_us,
      shared_best_name,
      (shared_best_us / separate_us - 1.0f) * 100.0f,
      merged_us,
      (merged_us / separate_us - 1.0f) * 100.0f,
      indirect_global_us,
      (indirect_global_us / separate_us - 1.0f) * 100.0f);

  CUDA_CHECK(cudaFree(device_key_ptrs));
  for (auto* key : host_keys) {
    CUDA_CHECK(cudaFree(key));
  }
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(merged_keys));
  CUDA_CHECK(cudaFree(input));
}

template <int BETA>
void run_suite(int length, int N, int iters) {
  run_case<1, BETA>(length, N, iters);
  run_case<2, BETA>(length, N, iters);
  run_case<4, BETA>(length, N, iters);
  run_case<8, BETA>(length, N, iters);
  run_case<16, BETA>(length, N, iters);
  run_case<32, BETA>(length, N, iters);
}

int main(int argc, char** argv) {
  int N = 1 << 16;
  int iters = 20;
  if (argc >= 2) {
    N = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    iters = std::atoi(argv[2]);
  }

  cudaDeviceProp prop{};
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("device=%s N=%d iters=%d\n", prop.name, N, iters);

  run_suite<1>(18, N, iters);
  run_suite<3>(41, N, iters);
  return 0;
}
