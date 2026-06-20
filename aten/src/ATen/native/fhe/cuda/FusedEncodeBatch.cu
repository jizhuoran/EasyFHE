#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <vector>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

namespace fhe::fused_encode {

static constexpr int kPreEncodeBlockSize = 256;
static constexpr int kPreEncodeSharedMaxSlots = 2048;
static constexpr int kPreEncodeMaxLargeTiles = 32;
static constexpr int64_t kMax64BitValue = 9223372036854775295LL;

template <typename scalar_t>
__device__ __forceinline__ c10::complex<double> to_complex_double(
    const scalar_t& value) {
  return c10::complex<double>(
      static_cast<double>(value.real()),
      static_cast<double>(value.imag()));
}

template <typename scalar_t>
__global__ void pre_encode_large_stage_kernel(
    const scalar_t* input,
    c10::complex<double>* workspace,
    const uint32_t* rot_group,
    const c10::complex<double>* ksi_pows,
    int64_t slots,
    int64_t M,
    int64_t tiles_per_row) {
  const int64_t tile_size = kPreEncodeSharedMaxSlots;
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= tile_size) {
    return;
  }
  const int64_t row = blockIdx.y;
  const int64_t row_base = row * slots;

  c10::complex<double> values[kPreEncodeMaxLargeTiles];
#pragma unroll
  for (int64_t t = 0; t < kPreEncodeMaxLargeTiles; ++t) {
    values[t] = c10::complex<double>(0.0, 0.0);
  }
  for (int64_t t = 0; t < tiles_per_row; ++t) {
    values[t] = to_complex_double(input[row_base + t * tile_size + r]);
  }

  for (int64_t tile_group_size = tiles_per_row; tile_group_size > 1;
       tile_group_size >>= 1) {
    const int64_t half_tiles = tile_group_size >> 1;
    const int64_t len_size = tile_size * tile_group_size;
    const int64_t len_q = len_size << 2;
    const int64_t gap = M / len_q;

    for (int64_t group_tile = 0; group_tile < tiles_per_row;
         group_tile += tile_group_size) {
      for (int64_t q_tile = 0; q_tile < half_tiles; ++q_tile) {
        const int64_t j = q_tile * tile_size + r;
        const uint32_t rot = rot_group[j] % static_cast<uint32_t>(len_q);
        const int64_t root_index = (len_q - rot) * gap;
        const auto left = values[group_tile + q_tile];
        const auto right = values[group_tile + q_tile + half_tiles];
        const auto root = ksi_pows[root_index];
        values[group_tile + q_tile] = left + right;
        values[group_tile + q_tile + half_tiles] = (left - right) * root;
      }
    }
  }

  for (int64_t t = 0; t < tiles_per_row; ++t) {
    workspace[row_base + t * tile_size + r] = values[t];
  }
}

template <typename scalar_t>
__global__ void pre_encode_stage1_shared_kernel(
    const scalar_t* input,
    const uint32_t* rot_group,
    const c10::complex<double>* ksi_pows,
    const uint32_t* bitrev,
    double* output,
    int64_t slots,
    int64_t M) {
  extern __shared__ unsigned char shared_bytes[];
  auto workspace = reinterpret_cast<c10::complex<double>*>(shared_bytes);
  const int64_t row = blockIdx.x;
  const int64_t row_base = row * slots;

  for (int64_t i = threadIdx.x; i < slots; i += blockDim.x) {
    workspace[i] = to_complex_double(input[row_base + i]);
  }
  __syncthreads();

  for (int64_t len_size = slots; len_size >= 2; len_size >>= 1) {
    const int64_t len_h = len_size >> 1;
    const int64_t len_q = len_size << 2;
    const int64_t gap = M / len_q;
    const int64_t total_work = slots >> 1;

    for (int64_t rem = threadIdx.x; rem < total_work; rem += blockDim.x) {
      const int64_t group = rem / len_h;
      const int64_t j = rem - group * len_h;
      const int64_t base = group * len_size + j;

      const uint32_t rot = rot_group[j] % static_cast<uint32_t>(len_q);
      const int64_t root_index = (len_q - rot) * gap;
      const auto left = workspace[base];
      const auto right = workspace[base + len_h];
      const auto root = ksi_pows[root_index];
      workspace[base] = left + right;
      workspace[base + len_h] = (left - right) * root;
    }
    __syncthreads();
  }

  const double inv_slots = 1.0 / static_cast<double>(slots);
  for (int64_t i = threadIdx.x; i < slots; i += blockDim.x) {
    const auto value = workspace[bitrev[i]] * inv_slots;
    const int64_t out_base = row * (2 * slots) + 2 * i;
    output[out_base] = value.real();
    output[out_base + 1] = value.imag();
  }
}

__global__ void pre_encode_stage1_tile_kernel(
    const c10::complex<double>* workspace_global,
    const uint32_t* rot_group,
    const c10::complex<double>* ksi_pows,
    const uint32_t* bitrev,
    double* output,
    int64_t slots,
    int64_t M) {
  extern __shared__ unsigned char shared_bytes[];
  auto workspace = reinterpret_cast<c10::complex<double>*>(shared_bytes);
  const int64_t tile_size = kPreEncodeSharedMaxSlots;
  const int64_t tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t tile_base = tile * tile_size;
  const int64_t row_base = row * slots;

  for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
    workspace[i] = workspace_global[row_base + tile_base + i];
  }
  __syncthreads();

  for (int64_t len_size = tile_size; len_size >= 2; len_size >>= 1) {
    const int64_t len_h = len_size >> 1;
    const int64_t len_q = len_size << 2;
    const int64_t gap = M / len_q;
    const int64_t total_work = tile_size >> 1;

    for (int64_t rem = threadIdx.x; rem < total_work; rem += blockDim.x) {
      const int64_t group = rem / len_h;
      const int64_t j = rem - group * len_h;
      const int64_t base = group * len_size + j;

      const uint32_t rot = rot_group[j] % static_cast<uint32_t>(len_q);
      const int64_t root_index = (len_q - rot) * gap;
      const auto left = workspace[base];
      const auto right = workspace[base + len_h];
      const auto root = ksi_pows[root_index];
      workspace[base] = left + right;
      workspace[base + len_h] = (left - right) * root;
    }
    __syncthreads();
  }

  const double inv_slots = 1.0 / static_cast<double>(slots);
  for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
    const int64_t source = tile_base + i;
    const int64_t output_index = bitrev[source];
    const auto value = workspace[i] * inv_slots;
    const int64_t out_base = row * (2 * slots) + 2 * output_index;
    output[out_base] = value.real();
    output[out_base + 1] = value.imag();
  }
}

__global__ void pre_encode_stage1_tile_partial4_kernel(
    c10::complex<double>* workspace_global,
    const uint32_t* rot_group,
    const c10::complex<double>* ksi_pows,
    int64_t slots,
    int64_t M) {
  extern __shared__ unsigned char shared_bytes[];
  auto workspace = reinterpret_cast<c10::complex<double>*>(shared_bytes);
  const int64_t tile_size = kPreEncodeSharedMaxSlots;
  const int64_t tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t tile_base = tile * tile_size;
  const int64_t row_base = row * slots;

  for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
    workspace[i] = workspace_global[row_base + tile_base + i];
  }
  __syncthreads();

  for (int64_t len_size = tile_size; len_size >= 8; len_size >>= 1) {
    const int64_t len_h = len_size >> 1;
    const int64_t len_q = len_size << 2;
    const int64_t gap = M / len_q;
    const int64_t total_work = tile_size >> 1;

    for (int64_t rem = threadIdx.x; rem < total_work; rem += blockDim.x) {
      const int64_t group = rem / len_h;
      const int64_t j = rem - group * len_h;
      const int64_t base = group * len_size + j;

      const uint32_t rot = rot_group[j] % static_cast<uint32_t>(len_q);
      const int64_t root_index = (len_q - rot) * gap;
      const auto left = workspace[base];
      const auto right = workspace[base + len_h];
      const auto root = ksi_pows[root_index];
      workspace[base] = left + right;
      workspace[base + len_h] = (left - right) * root;
    }
    __syncthreads();
  }

  for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
    workspace_global[row_base + tile_base + i] = workspace[i];
  }
}

__device__ __forceinline__ uint64_t fit_double_to_native(
    double value,
    double scaling_factor,
    int64_t big_value_half,
    uint64_t modulus,
    uint64_t max_int_diff,
    uint64_t barret_ratio,
    uint64_t barret_k) {
  int64_t rounded = llround(value * scaling_factor);
  rounded = (rounded < 0) ? (kMax64BitValue + rounded) : rounded;
  uint64_t reduced = static_cast<uint64_t>(rounded);
  barret_reduction_64_64(
      reduced, reduced, modulus, barret_ratio, barret_k);
  if (rounded > big_value_half) {
    reduced = sub_mod(reduced, max_int_diff, modulus);
  }
  return reduced;
}

__device__ __forceinline__ int64_t round_double_to_native_int(
    double value,
    double scaling_factor) {
  int64_t rounded = llround(value * scaling_factor);
  return (rounded < 0) ? (kMax64BitValue + rounded) : rounded;
}

__device__ __forceinline__ int64_t round_double_to_signed_int(
    double value,
    double scaling_factor) {
  return llround(value * scaling_factor);
}

__device__ __forceinline__ uint64_t reduce_rounded_to_native(
    int64_t rounded,
    int64_t big_value_half,
    uint64_t modulus,
    uint64_t max_int_diff,
    uint64_t barret_ratio,
    uint64_t barret_k) {
  uint64_t reduced = static_cast<uint64_t>(rounded);
  barret_reduction_64_64(
      reduced, reduced, modulus, barret_ratio, barret_k);
  if (rounded > big_value_half) {
    reduced = sub_mod(reduced, max_int_diff, modulus);
  }
  return reduced;
}

__device__ __forceinline__ uint64_t reduce_signed_rounded_to_native(
    int64_t rounded,
    uint64_t modulus,
    uint64_t barret_ratio,
    uint64_t barret_k) {
  const bool neg = rounded < 0;
  const uint64_t abs_rounded =
      neg ? static_cast<uint64_t>(-(rounded + 1)) + 1
          : static_cast<uint64_t>(rounded);
  uint64_t reduced = abs_rounded;
  if (reduced >= modulus) {
    barret_reduction_64_64(
        reduced, reduced, modulus, barret_ratio, barret_k);
  }
  if (neg && reduced != 0) {
    reduced = modulus - reduced;
  }
  return reduced;
}

__device__ __forceinline__ void fft2_complex(
    c10::complex<double>& left,
    c10::complex<double>& right,
    const c10::complex<double>& root) {
  const auto old_left = left;
  const auto old_right = right;
  left = old_left + old_right;
  right = (old_left - old_right) * root;
}

__device__ __forceinline__ void butt_ntt_local(
    uint64_t& a,
    uint64_t& b,
    const uint64_t w,
    const uint64_t w_,
    const uint64_t p,
    const uint64_t two_p) {
  uint64_t U = mul_and_reduce_shoup(b, w, w_, p);
  if (a >= two_p) {
    a -= two_p;
  }
  b = a + (two_p - U);
  a += U;
}

template <uint8_t radix>
__device__ __forceinline__ void local_ntt_radix(
    uint64_t& local0,
    uint64_t& local1,
    uint64_t& local2,
    uint64_t& local3,
    uint64_t& local4,
    uint64_t& local5,
    uint64_t& local6,
    uint64_t& local7,
    const uint32_t tw_off,
    const uint64_t* __restrict__ W,
    const uint64_t* __restrict__ W_,
    const uint64_t prime,
    const uint64_t two_p) {
  static_assert(radix == 2 || radix == 4 || radix == 8);
  if constexpr (radix >= 8) {
    const uint64_t w = W[tw_off];
    const uint64_t ws = W_[tw_off];
    butt_ntt_local(local0, local4, w, ws, prime, two_p);
    butt_ntt_local(local1, local5, w, ws, prime, two_p);
    butt_ntt_local(local2, local6, w, ws, prime, two_p);
    butt_ntt_local(local3, local7, w, ws, prime, two_p);
  }

  if constexpr (radix >= 4) {
    const uint32_t off = 2 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_[off + 1];
    butt_ntt_local(local0, local2, w0, ws0, prime, two_p);
    butt_ntt_local(local1, local3, w0, ws0, prime, two_p);
    butt_ntt_local(local4, local6, w1, ws1, prime, two_p);
    butt_ntt_local(local5, local7, w1, ws1, prime, two_p);
  }
  if constexpr (radix >= 2) {
    const uint32_t off = 4 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_[off + 1];
    const uint64_t w2 = W[off + 2];
    const uint64_t ws2 = W_[off + 2];
    const uint64_t w3 = W[off + 3];
    const uint64_t ws3 = W_[off + 3];
    butt_ntt_local(local0, local1, w0, ws0, prime, two_p);
    butt_ntt_local(local2, local3, w1, ws1, prime, two_p);
    butt_ntt_local(local4, local5, w2, ws2, prime, two_p);
    butt_ntt_local(local6, local7, w3, ws3, prime, two_p);
  }
}

__global__ void fft4_fit_ntt_phase1_kernel(
    const c10::complex<double>* __restrict__ workspace_global,
    uint64_t* __restrict__ out_ptr,
    const uint32_t* __restrict__ rot_group,
    const c10::complex<double>* __restrict__ ksi_pows,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ max_int_diffs,
    const uint64_t* __restrict__ barret_ratio,
    const uint64_t* __restrict__ barret_k,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup,
    int64_t slots,
    int64_t M,
    int64_t cur_limbs,
    int64_t N,
    double scaling_factor) {
  const int64_t phase1_block = blockIdx.x;
  const int64_t limb = blockIdx.y;
  const int64_t row = blockIdx.z;
  const int64_t group_id = threadIdx.x / 8;
  const int64_t lane_id = threadIdx.x & 7;
  const int64_t n_init = 8 * phase1_block + lane_id;

  const int64_t slot0 = group_id * 1024 + n_init;
  const int64_t source0 = __brev(static_cast<uint32_t>(slot0)) >> 17;
  const int64_t source_base = source0 & ~int64_t{3};
  const int64_t row_base = row * slots;

  c10::complex<double> v0 = workspace_global[row_base + source_base];
  c10::complex<double> v1 = workspace_global[row_base + source_base + 1];
  c10::complex<double> v2 = workspace_global[row_base + source_base + 2];
  c10::complex<double> v3 = workspace_global[row_base + source_base + 3];

  const int64_t gap4 = M / 16;
  const auto root4_0 =
      ksi_pows[(16 - (rot_group[0] % static_cast<uint32_t>(16))) * gap4];
  const auto root4_1 =
      ksi_pows[(16 - (rot_group[1] % static_cast<uint32_t>(16))) * gap4];
  fft2_complex(v0, v2, root4_0);
  fft2_complex(v1, v3, root4_1);

  const int64_t gap2 = M / 8;
  const auto root2 =
      ksi_pows[(8 - (rot_group[0] % static_cast<uint32_t>(8))) * gap2];
  fft2_complex(v0, v1, root2);
  fft2_complex(v2, v3, root2);

  const uint64_t prime = primes[limb];
  const uint64_t two_p = prime << 1;
  const uint64_t diff = max_int_diffs[limb];
  const uint64_t ratio = barret_ratio[limb];
  const uint64_t k = barret_k[limb];
  const int64_t big_half = kMax64BitValue >> 1;
  const double stage_scaling_factor =
      scaling_factor / static_cast<double>(slots);

  uint64_t local[8];
  local[0] = fit_double_to_native(
      v0.real(), stage_scaling_factor, big_half, prime, diff, ratio, k);
  local[1] = fit_double_to_native(
      v2.real(), stage_scaling_factor, big_half, prime, diff, ratio, k);
  local[2] = fit_double_to_native(
      v1.real(), stage_scaling_factor, big_half, prime, diff, ratio, k);
  local[3] = fit_double_to_native(
      v3.real(), stage_scaling_factor, big_half, prime, diff, ratio, k);
  local[4] = fit_double_to_native(
      v0.imag(), stage_scaling_factor, big_half, prime, diff, ratio, k);
  local[5] = fit_double_to_native(
      v2.imag(), stage_scaling_factor, big_half, prime, diff, ratio, k);
  local[6] = fit_double_to_native(
      v1.imag(), stage_scaling_factor, big_half, prime, diff, ratio, k);
  local[7] = fit_double_to_native(
      v3.imag(), stage_scaling_factor, big_half, prime, diff, ratio, k);

  const uint64_t* W = power_of_roots + limb * N;
  const uint64_t* W_shoup = power_of_roots_shoup + limb * N;
  local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      W,
      W_shoup,
      prime,
      two_p);

  __shared__ uint64_t transpose_matrix[8][8 + 1][8 + 1];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    transpose_matrix[lane_id][j][group_id] = local[j];
  }
  __syncthreads();

#pragma unroll
  for (int l = 0; l < 8; ++l) {
    local[l] = transpose_matrix[lane_id][group_id][l];
  }

  local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + group_id,
      W,
      W_shoup,
      prime,
      two_p);

  const int64_t out_base = row * cur_limbs * N + limb * N;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t coeff = group_id * 8192 + j * 1024 + n_init;
    out_ptr[out_base + coeff] = local[j];
  }
}

__global__ void fft4_round_ntt_phase1_all_limbs_kernel(
    const c10::complex<double>* __restrict__ workspace_global,
    uint64_t* __restrict__ out_ptr,
    const uint32_t* __restrict__ rot_group,
    const c10::complex<double>* __restrict__ ksi_pows,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ max_int_diffs,
    const uint64_t* __restrict__ barret_ratio,
    const uint64_t* __restrict__ barret_k,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup,
    int64_t slots,
    int64_t M,
    int64_t cur_limbs,
    int64_t N,
    double scaling_factor) {
  const int64_t phase1_block = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t group_id = threadIdx.x / 8;
  const int64_t lane_id = threadIdx.x & 7;
  const int64_t n_init = 8 * phase1_block + lane_id;

  const int64_t slot0 = group_id * 1024 + n_init;
  const int64_t source0 = __brev(static_cast<uint32_t>(slot0)) >> 17;
  const int64_t source_base = source0 & ~int64_t{3};
  const int64_t row_base = row * slots;

  c10::complex<double> v0 = workspace_global[row_base + source_base];
  c10::complex<double> v1 = workspace_global[row_base + source_base + 1];
  c10::complex<double> v2 = workspace_global[row_base + source_base + 2];
  c10::complex<double> v3 = workspace_global[row_base + source_base + 3];

  const int64_t gap4 = M / 16;
  const auto root4_0 =
      ksi_pows[(16 - (rot_group[0] % static_cast<uint32_t>(16))) * gap4];
  const auto root4_1 =
      ksi_pows[(16 - (rot_group[1] % static_cast<uint32_t>(16))) * gap4];
  fft2_complex(v0, v2, root4_0);
  fft2_complex(v1, v3, root4_1);

  const int64_t gap2 = M / 8;
  const auto root2 =
      ksi_pows[(8 - (rot_group[0] % static_cast<uint32_t>(8))) * gap2];
  fft2_complex(v0, v1, root2);
  fft2_complex(v2, v3, root2);

  const double stage_scaling_factor =
      scaling_factor / static_cast<double>(slots);
  const int64_t rounded[8] = {
      round_double_to_signed_int(v0.real(), stage_scaling_factor),
      round_double_to_signed_int(v2.real(), stage_scaling_factor),
      round_double_to_signed_int(v1.real(), stage_scaling_factor),
      round_double_to_signed_int(v3.real(), stage_scaling_factor),
      round_double_to_signed_int(v0.imag(), stage_scaling_factor),
      round_double_to_signed_int(v2.imag(), stage_scaling_factor),
      round_double_to_signed_int(v1.imag(), stage_scaling_factor),
      round_double_to_signed_int(v3.imag(), stage_scaling_factor),
  };

  __shared__ uint64_t transpose_matrix[8][8 + 1][8 + 1];
  for (int64_t limb = 0; limb < cur_limbs; ++limb) {
    const uint64_t prime = primes[limb];
    const uint64_t two_p = prime << 1;
    const uint64_t ratio = barret_ratio[limb];
    const uint64_t k = barret_k[limb];

    uint64_t local[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      local[i] = reduce_signed_rounded_to_native(rounded[i], prime, ratio, k);
    }

    const uint64_t* W = power_of_roots + limb * N;
    const uint64_t* W_shoup = power_of_roots_shoup + limb * N;
    local_ntt_radix<8>(
        local[0],
        local[1],
        local[2],
        local[3],
        local[4],
        local[5],
        local[6],
        local[7],
        1,
        W,
        W_shoup,
        prime,
        two_p);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      transpose_matrix[lane_id][j][group_id] = local[j];
    }
    __syncthreads();

#pragma unroll
    for (int l = 0; l < 8; ++l) {
      local[l] = transpose_matrix[lane_id][group_id][l];
    }
    __syncthreads();

    local_ntt_radix<8>(
        local[0],
        local[1],
        local[2],
        local[3],
        local[4],
        local[5],
        local[6],
        local[7],
        8 + group_id,
        W,
        W_shoup,
        prime,
        two_p);

    const int64_t out_base = row * cur_limbs * N + limb * N;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t coeff = group_id * 8192 + j * 1024 + n_init;
      out_ptr[out_base + coeff] = local[j];
    }
  }
}

template <int LIMB_CHUNK>
__global__ void fft4_round_ntt_phase1_limb_chunk_kernel(
    const c10::complex<double>* __restrict__ workspace_global,
    uint64_t* __restrict__ out_ptr,
    const uint32_t* __restrict__ rot_group,
    const c10::complex<double>* __restrict__ ksi_pows,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ max_int_diffs,
    const uint64_t* __restrict__ barret_ratio,
    const uint64_t* __restrict__ barret_k,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup,
    int64_t slots,
    int64_t M,
    int64_t cur_limbs,
    int64_t N,
    double scaling_factor) {
  (void)max_int_diffs;
  const int64_t phase1_block = blockIdx.x;
  const int64_t limb = blockIdx.y * LIMB_CHUNK + threadIdx.y;
  const int64_t row = blockIdx.z;
  const int64_t group_id = threadIdx.x / 8;
  const int64_t lane_id = threadIdx.x & 7;
  const int64_t n_init = 8 * phase1_block + lane_id;

  __shared__ int64_t rounded_s[8][64 + 1];
  __shared__ uint64_t transpose_matrix[LIMB_CHUNK][8][8 + 1][8 + 1];

  if (threadIdx.y == 0) {
    const int64_t slot0 = group_id * 1024 + n_init;
    const int64_t source0 = __brev(static_cast<uint32_t>(slot0)) >> 17;
    const int64_t source_base = source0 & ~int64_t{3};
    const int64_t row_base = row * slots;

    c10::complex<double> v0 = workspace_global[row_base + source_base];
    c10::complex<double> v1 = workspace_global[row_base + source_base + 1];
    c10::complex<double> v2 = workspace_global[row_base + source_base + 2];
    c10::complex<double> v3 = workspace_global[row_base + source_base + 3];

    const int64_t gap4 = M / 16;
    const auto root4_0 =
        ksi_pows[(16 - (rot_group[0] % static_cast<uint32_t>(16))) * gap4];
    const auto root4_1 =
        ksi_pows[(16 - (rot_group[1] % static_cast<uint32_t>(16))) * gap4];
    fft2_complex(v0, v2, root4_0);
    fft2_complex(v1, v3, root4_1);

    const int64_t gap2 = M / 8;
    const auto root2 =
        ksi_pows[(8 - (rot_group[0] % static_cast<uint32_t>(8))) * gap2];
    fft2_complex(v0, v1, root2);
    fft2_complex(v2, v3, root2);

    const double stage_scaling_factor =
        scaling_factor / static_cast<double>(slots);
    rounded_s[0][threadIdx.x] =
        round_double_to_signed_int(v0.real(), stage_scaling_factor);
    rounded_s[1][threadIdx.x] =
        round_double_to_signed_int(v2.real(), stage_scaling_factor);
    rounded_s[2][threadIdx.x] =
        round_double_to_signed_int(v1.real(), stage_scaling_factor);
    rounded_s[3][threadIdx.x] =
        round_double_to_signed_int(v3.real(), stage_scaling_factor);
    rounded_s[4][threadIdx.x] =
        round_double_to_signed_int(v0.imag(), stage_scaling_factor);
    rounded_s[5][threadIdx.x] =
        round_double_to_signed_int(v2.imag(), stage_scaling_factor);
    rounded_s[6][threadIdx.x] =
        round_double_to_signed_int(v1.imag(), stage_scaling_factor);
    rounded_s[7][threadIdx.x] =
        round_double_to_signed_int(v3.imag(), stage_scaling_factor);
  }
  __syncthreads();

  const bool valid_limb = limb < cur_limbs;
  uint64_t local[8];
  if (valid_limb) {
    const uint64_t prime = primes[limb];
    const uint64_t two_p = prime << 1;
    const uint64_t ratio = barret_ratio[limb];
    const uint64_t k = barret_k[limb];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      local[i] = reduce_signed_rounded_to_native(
          rounded_s[i][threadIdx.x], prime, ratio, k);
    }

    const uint64_t* W = power_of_roots + limb * N;
    const uint64_t* W_shoup = power_of_roots_shoup + limb * N;
    local_ntt_radix<8>(
        local[0],
        local[1],
        local[2],
        local[3],
        local[4],
        local[5],
        local[6],
        local[7],
        1,
        W,
        W_shoup,
        prime,
        two_p);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      transpose_matrix[threadIdx.y][lane_id][j][group_id] = local[j];
    }
  }
  __syncthreads();

  if (valid_limb) {
    const uint64_t prime = primes[limb];
    const uint64_t two_p = prime << 1;
    const uint64_t* W = power_of_roots + limb * N;
    const uint64_t* W_shoup = power_of_roots_shoup + limb * N;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      local[l] = transpose_matrix[threadIdx.y][lane_id][group_id][l];
    }

    local_ntt_radix<8>(
        local[0],
        local[1],
        local[2],
        local[3],
        local[4],
        local[5],
        local[6],
        local[7],
        8 + group_id,
        W,
        W_shoup,
        prime,
        two_p);

    const int64_t out_base = row * cur_limbs * N + limb * N;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t coeff = group_id * 8192 + j * 1024 + n_init;
      out_ptr[out_base + coeff] = local[j];
    }
  }
}

template <int NUM_ROUNDS>
__device__ __forceinline__ void warp_butterfly(
    uint64_t& i1,
    uint64_t& i2,
    uint32_t& stage_off,
    const uint32_t laneID,
    const uint64_t* __restrict__ base_inv,
    const uint64_t* __restrict__ base_inv_,
    const uint64_t prime,
    const uint64_t two_p) {
  static_assert(NUM_ROUNDS >= 2);
  butt_ntt_local(
      i1, i2, base_inv[stage_off], base_inv_[stage_off], prime, two_p);

#pragma unroll
  for (int shift = NUM_ROUNDS - 2; shift >= 0; --shift) {
    const uint32_t offset = 1u << shift;
    const bool lower_half = (laneID & offset) == 0;
    auto tmp = lower_half ? i2 : i1;
    tmp = __shfl_xor_sync(0xFFFFFFFF, tmp, offset);
    if (lower_half) {
      i2 = tmp;
    } else {
      i1 = tmp;
    }
    stage_off <<= 1;
    const uint32_t idx = stage_off + (laneID >> shift);
    butt_ntt_local(i1, i2, base_inv[idx], base_inv_[idx], prime, two_p);
  }
}

template <int SMEM_PAD>
__global__ void ntt65536_phase2_kernel(
    uint64_t* __restrict__ inout_ptr,
    const size_t LN,
    const uint64_t* __restrict__ base_inv,
    const uint64_t* __restrict__ base_inv_,
    const uint64_t* __restrict__ primes) {
  constexpr size_t LOG_N = 16;
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t LOG_RADIX = LOG_N - 6;
  constexpr int R1_RADIX = 64;
  constexpr int DATA_SIZE = 1u << LOG_RADIX;
  constexpr int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);
  constexpr int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;
  constexpr int SMEM_STRIDE = WARP_SIZE + SMEM_PAD;

  const uint32_t row = blockIdx.z;
  const uint32_t limb = blockIdx.y;
  inout_ptr += row * LN;
  base_inv += limb * N;
  base_inv_ += limb * N;
  const uint64_t prime = primes[limb];
  const uint64_t two_p = prime << 1;

  auto g_row = reinterpret_cast<uint64_t(*)[R1_RADIX][DATA_SIZE]>(inout_ptr);
  __shared__ uint64_t tile[2][NUM_WARP][SMEM_STRIDE];

  uint32_t stage_off = R1_RADIX + blockIdx.x;
  uint64_t i1 = g_row[limb][blockIdx.x][threadIdx.x];
  uint64_t i2 = g_row[limb][blockIdx.x][threadIdx.x + kBLOCK_SIZE];

  tile[0][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i1;
  tile[1][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i2;
  __syncthreads();

  uint32_t laneID = threadIdx.x % WARP_SIZE;
  uint32_t groupID = threadIdx.x / WARP_SIZE;
  i1 = tile[0][groupID][laneID];
  i2 = tile[1][groupID][laneID];

  warp_butterfly<6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  tile[0][groupID][laneID] = i1;
  tile[1][groupID][laneID] = i2;
  __syncthreads();

  constexpr int SECOND_GROUP_SIZE = DATA_SIZE / WARP_SIZE / 4;
  laneID = threadIdx.x % SECOND_GROUP_SIZE;
  groupID = threadIdx.x / SECOND_GROUP_SIZE;

  const uint32_t half_group = groupID >> 1;
  if ((groupID & 1) == 0) {
    i1 = tile[0][laneID][half_group];
    i2 = tile[0][laneID + SECOND_GROUP_SIZE][half_group];
  } else {
    i1 = tile[1][laneID][half_group];
    i2 = tile[1][laneID + SECOND_GROUP_SIZE][half_group];
  }

  stage_off = stage_off * 2 + groupID;
  warp_butterfly<LOG_RADIX - 6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  for (int k = 0; k < 3; k++) {
    if (i1 >= prime) {
      i1 -= prime;
    }
    if (i2 >= prime) {
      i2 -= prime;
    }
  }

  auto g_row_out =
      reinterpret_cast<ulonglong2(*)[R1_RADIX][DATA_SIZE / 2]>(inout_ptr);
  ulonglong2 i12 = {i1, i2};
  g_row_out[limb][blockIdx.x][threadIdx.x] = i12;
}

template <typename DDTYPE>
__global__ void fit_to_native_vector_kernel(
    DDTYPE* in_ptr,
    double scaling_factor,
    int64_t bigValueHf,
    uint64_t* out_ptr,
    uint64_t* native_modulus,
    uint64_t* max_int_diffs_ptr,
    uint64_t* barret_ratio_ptr,
    uint64_t* barret_k_ptr,
    int64_t N,
    const size_t L_OUTN,
    const size_t L_INN,
    int64_t slots,
    int64_t gap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  auto cipher_id = blockIdx.z;
  in_ptr += cipher_id * L_INN;
  out_ptr += cipher_id * L_OUTN;

  if (i < slots) {
    const int l = blockIdx.y;
    int64_t diff = max_int_diffs_ptr[l];
    int64_t re = llround(in_ptr[2 * i] * scaling_factor);
    int64_t im = llround(in_ptr[2 * i + 1] * scaling_factor);

    re = (re < 0) ? (kMax64BitValue + re) : re;
    im = (im < 0) ? (kMax64BitValue + im) : im;

    uint64_t re_ = re;
    uint64_t im_ = im;

    barret_reduction_64_64(
        re_, re_, native_modulus[l], barret_ratio_ptr[l], barret_k_ptr[l]);
    barret_reduction_64_64(
        im_, im_, native_modulus[l], barret_ratio_ptr[l], barret_k_ptr[l]);

    if (re > bigValueHf) {
      re_ = sub_mod(re_, diff, native_modulus[l]);
    }
    if (im > bigValueHf) {
      im_ = sub_mod(im_, diff, native_modulus[l]);
    }

    out_ptr[l * N + gap * i] = re_;
    out_ptr[l * N + gap * (i + slots)] = im_;
  }
}

} // namespace fhe::fused_encode

namespace at::native {

static bool launch_fused_encode_phase1_experimental(
    const Tensor& workspace,
    Tensor& out,
    const Tensor& rotGroup,
    const Tensor& ksiPows,
    const Tensor& primes,
    const Tensor& max_int_diffs,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup,
    int64_t slots,
    int64_t M,
    int64_t cur_limbs,
    int64_t N,
    double scaling_factor,
    at::cuda::CUDAStream stream) {
  const char* chunk_env = std::getenv("EASYFHE_FUSED_ENCODE_PHASE1_LIMB_CHUNK");
  if (chunk_env != nullptr) {
    const int chunk = std::atoi(chunk_env);
    TORCH_CHECK(
        chunk == 2 || chunk == 3 || chunk == 4 || chunk == 6,
        "EASYFHE_FUSED_ENCODE_PHASE1_LIMB_CHUNK must be one of 2, 3, 4, or 6");
    const dim3 grid(N / 512, (cur_limbs + chunk - 1) / chunk, out.size(0));
    auto launch_chunk = [&](auto chunk_constant) {
      constexpr int kChunk = decltype(chunk_constant)::value;
      fhe::fused_encode::fft4_round_ntt_phase1_limb_chunk_kernel<kChunk><<<
          grid,
          dim3(64, kChunk),
          0,
          stream>>>(
          workspace.data_ptr<c10::complex<double>>(),
          out.data_ptr<uint64_t>(),
          rotGroup.data_ptr<uint32_t>(),
          ksiPows.data_ptr<c10::complex<double>>(),
          primes.data_ptr<uint64_t>(),
          max_int_diffs.data_ptr<uint64_t>(),
          barret_ratio.data_ptr<uint64_t>(),
          barret_k.data_ptr<uint64_t>(),
          power_of_roots.data_ptr<uint64_t>(),
          power_of_roots_shoup.data_ptr<uint64_t>(),
          slots,
          M,
          cur_limbs,
          N,
          scaling_factor);
    };
    if (chunk == 2) {
      launch_chunk(std::integral_constant<int, 2>{});
    } else if (chunk == 3) {
      launch_chunk(std::integral_constant<int, 3>{});
    } else if (chunk == 4) {
      launch_chunk(std::integral_constant<int, 4>{});
    } else if (chunk == 6) {
      launch_chunk(std::integral_constant<int, 6>{});
    }
    return true;
  }

  if (std::getenv("EASYFHE_FUSED_ENCODE_PHASE1_LOOP_LIMBS") != nullptr) {
    fhe::fused_encode::fft4_round_ntt_phase1_all_limbs_kernel<<<
        dim3(N / 512, out.size(0)),
        64,
        0,
        stream>>>(
        workspace.data_ptr<c10::complex<double>>(),
        out.data_ptr<uint64_t>(),
        rotGroup.data_ptr<uint32_t>(),
        ksiPows.data_ptr<c10::complex<double>>(),
        primes.data_ptr<uint64_t>(),
        max_int_diffs.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        slots,
        M,
        cur_limbs,
        N,
        scaling_factor);
    return true;
  }

  return false;
}

static void launch_fused_encode_ntt2_phase2(
    Tensor& values,
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup,
    const Tensor& primes,
    int64_t cur_limbs,
    int64_t N,
    int64_t batch_size,
    at::cuda::CUDAStream stream) {
  constexpr int kRadix = 64;
  const dim3 grid(kRadix, cur_limbs, batch_size);
  const dim3 block(N / kRadix / 2);
  const char* pad_env = std::getenv("EASYFHE_FUSED_ENCODE_NTT2_SMEM_PAD");
  const int pad = pad_env == nullptr ? 1 : std::atoi(pad_env);

  auto launch_pad = [&](auto pad_constant) {
    constexpr int kPad = decltype(pad_constant)::value;
    fhe::fused_encode::ntt65536_phase2_kernel<kPad><<<grid, block, 0, stream>>>(
        values.data_ptr<uint64_t>(),
        cur_limbs * N,
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>());
  };

  if (pad == 1) {
    launch_pad(std::integral_constant<int, 1>{});
  } else if (pad == 2) {
    launch_pad(std::integral_constant<int, 2>{});
  } else if (pad == 6) {
    launch_pad(std::integral_constant<int, 6>{});
  } else {
    TORCH_CHECK(false, "EASYFHE_FUSED_ENCODE_NTT2_SMEM_PAD must be one of 1, 2, or 6");
  }
}

[[maybe_unused]] static Tensor fused_encode_batch_impl_cuda(
    const Tensor& raw,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    int64_t M,
    double scaling_factor,
    const Tensor& rotGroup,
    const Tensor& ksiPows,
    const Tensor& primes,
    const Tensor& max_int_diffs,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  TORCH_INTERNAL_ASSERT(raw.dim() == 2);
  TORCH_CHECK(slots == 32768, "fused_encode_batch expects slots=32768");
  TORCH_CHECK(N == 65536, "fused_encode_batch expects N=65536");
  Tensor input_2d = raw.contiguous();
  const int64_t batch_size = input_2d.size(0);
  const int64_t tiles_per_row = slots / fhe::fused_encode::kPreEncodeSharedMaxSlots;
  TORCH_INTERNAL_ASSERT(tiles_per_row == 16);

  Tensor workspace = at::empty(
      {batch_size, slots},
      input_2d.options().dtype(at::kComplexDouble));
  Tensor out = at::empty({batch_size, cur_limbs, N}, primes.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_COMPLEX_TYPES_AND(
      at::ScalarType::ComplexHalf,
      input_2d.scalar_type(),
      "fused_encode_batch_impl_cuda",
      [&] {
        fhe::fused_encode::pre_encode_large_stage_kernel<<<
            dim3(
                (fhe::fused_encode::kPreEncodeSharedMaxSlots +
                 fhe::fused_encode::kPreEncodeBlockSize - 1) /
                    fhe::fused_encode::kPreEncodeBlockSize,
                batch_size),
            fhe::fused_encode::kPreEncodeBlockSize,
            0,
            stream>>>(
            input_2d.data_ptr<scalar_t>(),
            workspace.data_ptr<c10::complex<double>>(),
            rotGroup.data_ptr<uint32_t>(),
            ksiPows.data_ptr<c10::complex<double>>(),
            slots,
            M,
            tiles_per_row);
      });

  const size_t shared_bytes =
      static_cast<size_t>(fhe::fused_encode::kPreEncodeSharedMaxSlots) *
      sizeof(c10::complex<double>);
  fhe::fused_encode::pre_encode_stage1_tile_partial4_kernel<<<
      dim3(tiles_per_row, batch_size),
      fhe::fused_encode::kPreEncodeBlockSize,
      shared_bytes,
      stream>>>(
      workspace.data_ptr<c10::complex<double>>(),
      rotGroup.data_ptr<uint32_t>(),
      ksiPows.data_ptr<c10::complex<double>>(),
      slots,
      M);

  if (!launch_fused_encode_phase1_experimental(
          workspace,
          out,
          rotGroup,
          ksiPows,
          primes,
          max_int_diffs,
          barret_ratio,
          barret_k,
          power_of_roots,
          power_of_roots_shoup,
          slots,
          M,
          cur_limbs,
          N,
          scaling_factor,
          stream)) {
    fhe::fused_encode::fft4_fit_ntt_phase1_kernel<<<
        dim3(N / 512, cur_limbs, batch_size),
        64,
        0,
        stream>>>(
        workspace.data_ptr<c10::complex<double>>(),
        out.data_ptr<uint64_t>(),
        rotGroup.data_ptr<uint32_t>(),
        ksiPows.data_ptr<c10::complex<double>>(),
        primes.data_ptr<uint64_t>(),
        max_int_diffs.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        slots,
        M,
        cur_limbs,
        N,
        scaling_factor);
  }

  launch_fused_encode_ntt2_phase2(
      out,
      power_of_roots,
      power_of_roots_shoup,
      primes,
      cur_limbs,
      N,
      batch_size,
      stream);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor fused_encode_batch_cuda(
    const Tensor& packed,
    int64_t cur_limbs,
    int64_t slots,
    int64_t N,
    int64_t M,
    double scaling_factor,
    const Tensor& rotGroup,
    const Tensor& ksiPows,
    const Tensor& primes,
    const Tensor& max_int_diffs,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  TORCH_CHECK(packed.is_cuda(), "fused_encode_batch expects CUDA packed input");
  TORCH_CHECK(
      packed.scalar_type() == at::kComplexHalf ||
          packed.scalar_type() == at::kComplexFloat ||
          packed.scalar_type() == at::kComplexDouble,
      "fused_encode_batch expects complex32, complex64, or complex128 input");
  TORCH_CHECK(packed.dim() == 2, "fused_encode_batch expects [batch, slots]");
  TORCH_CHECK(packed.size(1) == slots, "packed input slots mismatch");
  return fused_encode_batch_impl_cuda(
      packed,
      N,
      cur_limbs,
      slots,
      M,
      scaling_factor,
      rotGroup,
      ksiPows,
      primes,
      max_int_diffs,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots);
}

} // namespace at::native
