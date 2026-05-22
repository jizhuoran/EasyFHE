#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>

#include <limits>
#include <optional>
#include <vector>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

enum class HMulPostOp : int {
  None = 0,
  AddCipher = 1,
  SubCipher = 2,
  AddScalar = 3,
  AddPlain = 4,
};

__global__ void hmul_raw_kernel(
    uint64_t* __restrict__ raw,
    const uint64_t* __restrict__ c0,
    const uint64_t* __restrict__ c1,
    const uint64_t* __restrict__ d0,
    const uint64_t* __restrict__ d1,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ q_mu,
    const int curr_limbs,
    const int N) {
  const int coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int limb = blockIdx.y;
  const int idx = limb * N + coeff;
  const int LN = curr_limbs * N;

  const uint64_t prime = primes[limb];
  const uint64_t mu0 = q_mu[limb * 2];
  const uint64_t mu1 = q_mu[limb * 2 + 1];

  const uint64_t a0 = c0[idx];
  const uint64_t a1 = c1[idx];
  const uint64_t b0 = d0[idx];
  const uint64_t b1 = d1[idx];

  const uint64_t bx = mul_mod(a0, b0, prime, mu0, mu1);
  uint64_t ax = mul_mod(a0, b1, prime, mu0, mu1);
  const uint64_t ax_other = mul_mod(a1, b0, prime, mu0, mu1);
  ax = add_mod(ax, ax_other, prime);
  const uint64_t axax = mul_mod(a1, b1, prime, mu0, mu1);

  raw[idx] = bx;
  raw[LN + idx] = ax;
  raw[2 * LN + idx] = axax;
}

template <bool APPLY_DOUBLE>
__global__ void hmul_relin_scale_last_limb_kernel(
    uint64_t* __restrict__ out_last,
    const uint64_t* __restrict__ raw,
    const uint64_t* __restrict__ inner_product,
    const uint64_t* __restrict__ moddown_base,
    const uint64_t* __restrict__ prod_inv,
    const uint64_t* __restrict__ prod_inv_shoup,
    const uint64_t* __restrict__ primes,
    const int curr_limbs,
    const int sizeP,
    const int N) {
  const int coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int cv = blockIdx.y;
  const int limb = curr_limbs - 1;
  const int idx = limb * N + coeff;
  const int LN = curr_limbs * N;
  const int L_INN = (curr_limbs + sizeP) * N;
  const uint64_t prime = primes[limb];

  uint64_t key = sub_mod(
      inner_product[cv * L_INN + idx],
      moddown_base[cv * L_INN + idx],
      prime);
  key = mul_and_reduce_shoup(key, prod_inv[limb], prod_inv_shoup[limb], prime);
  if (key >= prime) {
    key -= prime;
  }

  uint64_t value = add_mod(raw[cv * LN + idx], key, prime);
  if constexpr (APPLY_DOUBLE) {
    value = add_mod(value, value, prime);
  }
  out_last[cv * N + coeff] = value;
}

__device__ __forceinline__ uint64_t hmul_reduce_lazy_4p(
    uint64_t x,
    const uint64_t p,
    const uint64_t two_p) {
  if (x >= two_p) {
    x -= two_p;
  }
  if (x >= p) {
    x -= p;
  }
  return x;
}

__device__ __forceinline__ void hmul_butt_ntt_local(
    uint64_t& a,
    uint64_t& b,
    const uint64_t w,
    const uint64_t w_shoup,
    const uint64_t p,
    const uint64_t two_p) {
  uint64_t u = mul_and_reduce_shoup(b, w, w_shoup, p);
  if (a >= two_p) {
    a -= two_p;
  }
  b = a + (two_p - u);
  a += u;
}

template <uint8_t radix>
__device__ __forceinline__ void hmul_local_ntt_radix(
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
    const uint64_t* __restrict__ W_shoup,
    const uint64_t prime,
    const uint64_t two_p) {
  static_assert(radix == 2 || radix == 4 || radix == 8);
  if constexpr (radix >= 8) {
    const uint64_t w = W[tw_off];
    const uint64_t ws = W_shoup[tw_off];
    hmul_butt_ntt_local(local0, local4, w, ws, prime, two_p);
    hmul_butt_ntt_local(local1, local5, w, ws, prime, two_p);
    hmul_butt_ntt_local(local2, local6, w, ws, prime, two_p);
    hmul_butt_ntt_local(local3, local7, w, ws, prime, two_p);
  }

  if constexpr (radix >= 4) {
    const uint32_t off = 2 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_shoup[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_shoup[off + 1];
    hmul_butt_ntt_local(local0, local2, w0, ws0, prime, two_p);
    hmul_butt_ntt_local(local1, local3, w0, ws0, prime, two_p);
    hmul_butt_ntt_local(local4, local6, w1, ws1, prime, two_p);
    hmul_butt_ntt_local(local5, local7, w1, ws1, prime, two_p);
  }

  if constexpr (radix >= 2) {
    const uint32_t off = 4 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_shoup[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_shoup[off + 1];
    const uint64_t w2 = W[off + 2];
    const uint64_t ws2 = W_shoup[off + 2];
    const uint64_t w3 = W[off + 3];
    const uint64_t ws3 = W_shoup[off + 3];
    hmul_butt_ntt_local(local0, local1, w0, ws0, prime, two_p);
    hmul_butt_ntt_local(local2, local3, w1, ws1, prime, two_p);
    hmul_butt_ntt_local(local4, local5, w2, ws2, prime, two_p);
    hmul_butt_ntt_local(local6, local7, w3, ws3, prime, two_p);
  }
}

template <int NUM_ROUNDS>
__device__ __forceinline__ void hmul_warp_butterfly(
    uint64_t& i1,
    uint64_t& i2,
    uint32_t& stage_off,
    const uint32_t laneID,
    const uint64_t* __restrict__ W,
    const uint64_t* __restrict__ W_shoup,
    const uint64_t prime,
    const uint64_t two_p) {
  static_assert(NUM_ROUNDS >= 2);
  hmul_butt_ntt_local(i1, i2, W[stage_off], W_shoup[stage_off], prime, two_p);

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
    hmul_butt_ntt_local(i1, i2, W[idx], W_shoup[idx], prime, two_p);
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void hmul_moddown_base_convert_ntt_phase1_kernel(
    uint64_t* __restrict__ workspace,
    const int64_t L_IN,
    const int64_t curr_limbs,
    const int64_t sizeP,
    const uint64_t* __restrict__ hat_mod_end,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barret_ratios,
    const uint64_t* __restrict__ barret_ks,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup) {
  static_assert(NUM_GROUPS == 8);
  constexpr int GROUP_SIZE = 8;
  constexpr int C_N = 1 << LOG_N;
  constexpr int COEFF_STRIDE = C_N / (8 * GROUP_SIZE);

  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int64_t limb = blockIdx.y;
  const int64_t cv_id = blockIdx.z;
  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  extern __shared__ uint64_t hat_mod_end_shared[];
  for (int t = threadIdx.x; t < sizeP; t += blockDim.x) {
    hat_mod_end_shared[t] = hat_mod_end[t + limb * sizeP];
  }
  __syncthreads();

  uint64_t* cv_workspace = workspace + cv_id * L_IN * C_N;
  const uint64_t* p_limbs = cv_workspace + curr_limbs * C_N;
  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][COEFF_STRIDE]>(cv_workspace);

  power_of_roots += limb * C_N;
  power_of_roots_shoup += limb * C_N;
  const uint64_t prime = primes[limb];
  const uint64_t two_p = prime << 1;
  const uint64_t barret_ratio = barret_ratios[limb];
  const uint64_t barret_k = barret_ks[limb];

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int coeff = j * (GROUP_SIZE * COEFF_STRIDE) +
        groupID * COEFF_STRIDE + N_init;
    uint128_t accum{0};
    for (int i = 0; i < sizeP; i++) {
      const uint64_t op1 = p_limbs[i * C_N + coeff];
      const uint64_t op2 = hat_mod_end_shared[i];
      uint128_t out = mult_64_64_128(op1, op2);
      inplace_add_128_128(out, accum);
    }
    local[j] = barret_reduction_128_64(accum, prime, barret_ratio, barret_k);
  }

  hmul_local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p);

  __shared__ uint64_t transpose_matrix[NUM_GROUPS][GROUP_SIZE + 1][8 + 1];

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    transpose_matrix[laneID][j][groupID] = local[j];
  }
  __syncthreads();

#pragma unroll
  for (int l = 0; l < 8; ++l) {
    local[l] = transpose_matrix[laneID][groupID][l];
  }

  hmul_local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[limb][groupID][j][N_init] = local[j];
  }
}

template <size_t LOG_N, int NUM_WARP>
__device__ __forceinline__ ulonglong2 hmul_ntt_phase2_pair(
    uint64_t* __restrict__ inout_ptr,
    const uint32_t poly_idx,
    const uint32_t block_x,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup,
    const uint64_t prime,
    const uint64_t two_p,
    uint64_t (&tile)[2][NUM_WARP][WARP_SIZE + 1]) {
  constexpr size_t LOG_RADIX = LOG_N - 6;
  constexpr int R1_RADIX = 64;
  constexpr int DATA_SIZE = 1u << LOG_RADIX;
  constexpr int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);

  auto g_row = reinterpret_cast<uint64_t(*)[R1_RADIX][DATA_SIZE]>(inout_ptr);
  uint32_t stage_off = R1_RADIX + block_x;

  uint64_t i1 = g_row[poly_idx][block_x][threadIdx.x];
  uint64_t i2 = g_row[poly_idx][block_x][threadIdx.x + kBLOCK_SIZE];

  tile[0][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i1;
  tile[1][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i2;
  __syncthreads();

  uint32_t laneID = threadIdx.x % WARP_SIZE;
  uint32_t groupID = threadIdx.x / WARP_SIZE;
  i1 = tile[0][groupID][laneID];
  i2 = tile[1][groupID][laneID];

  hmul_warp_butterfly<6>(
      i1,
      i2,
      stage_off,
      laneID,
      power_of_roots,
      power_of_roots_shoup,
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
  hmul_warp_butterfly<LOG_RADIX - 6>(
      i1,
      i2,
      stage_off,
      laneID,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p);

  i1 = hmul_reduce_lazy_4p(i1, prime, two_p);
  i2 = hmul_reduce_lazy_4p(i2, prime, two_p);
  return {i1, i2};
}

template <size_t LOG_N>
__global__ void hmul_ntt_phase2_store_kernel(
    uint64_t* __restrict__ workspace,
    const int64_t L_IN,
    const int64_t curr_limbs,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup) {
  constexpr size_t LOG_RADIX = LOG_N - 6;
  constexpr int DATA_SIZE = 1u << LOG_RADIX;
  constexpr int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);
  constexpr int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;
  constexpr int N = 1 << LOG_N;

  const uint32_t limb = blockIdx.y;
  const uint32_t cv_id = blockIdx.z;
  power_of_roots += limb * N;
  power_of_roots_shoup += limb * N;
  const uint64_t prime = primes[limb];
  const uint64_t two_p = prime << 1;
  __shared__ uint64_t tile[2][NUM_WARP][WARP_SIZE + 1];

  uint64_t* cv_workspace = workspace + cv_id * L_IN * N;
  const auto values = hmul_ntt_phase2_pair<LOG_N, NUM_WARP>(
      cv_workspace,
      limb,
      blockIdx.x,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p,
      tile);

  const int64_t src_base = blockIdx.x * DATA_SIZE + 2 * threadIdx.x;
  cv_workspace[limb * N + src_base] = values.x;
  cv_workspace[limb * N + src_base + 1] = values.y;
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void hmul_switch_const_mult_ntt_phase1_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ from_last,
    const uint64_t* __restrict__ cnst,
    const uint64_t* __restrict__ cnst_shoup,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ diffs,
    const uint64_t old_modulus_by_two,
    const int curr_limbs,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup) {
  static_assert(NUM_GROUPS == 8);
  constexpr int GROUP_SIZE = 8;
  constexpr int C_N = 1 << LOG_N;
  constexpr int COEFF_STRIDE = C_N / (8 * GROUP_SIZE);

  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int limb = blockIdx.y;
  const int cv_id = blockIdx.z;
  const int end_length = curr_limbs - 1;
  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  const uint64_t prime = primes[limb];
  const uint64_t two_p = prime << 1;
  power_of_roots += limb * C_N;
  power_of_roots_shoup += limb * C_N;

  const uint64_t* from_cv = from_last + cv_id * curr_limbs * C_N;
  uint64_t* out_cv = out + cv_id * end_length * C_N;
  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][COEFF_STRIDE]>(out_cv);

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int coeff = j * (GROUP_SIZE * COEFF_STRIDE) +
        groupID * COEFF_STRIDE + N_init;
    const uint64_t in_val = from_cv[coeff];
    uint64_t switched =
        in_val + (in_val > old_modulus_by_two ? diffs[limb] : uint64_t{0});
    if (switched >= prime) {
      switched -= prime;
    }

    uint64_t scaled =
        mul_and_reduce_shoup(switched, cnst[limb], cnst_shoup[limb], prime);
    if (scaled >= prime) {
      scaled -= prime;
    }
    local[j] = scaled;
  }

  hmul_local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p);

  __shared__ uint64_t transpose_matrix[NUM_GROUPS][GROUP_SIZE + 1][8 + 1];

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    transpose_matrix[laneID][j][groupID] = local[j];
  }
  __syncthreads();

#pragma unroll
  for (int l = 0; l < 8; ++l) {
    local[l] = transpose_matrix[laneID][groupID][l];
  }

  hmul_local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[limb][groupID][j][N_init] = local[j];
  }
}

template <size_t LOG_N, bool APPLY_DOUBLE, HMulPostOp POST_OP>
__global__ void hmul_ntt_phase2_finalize_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ raw,
    const uint64_t* __restrict__ inner_product,
    const uint64_t* __restrict__ moddown_base,
    const uint64_t* __restrict__ prod_inv,
    const uint64_t* __restrict__ prod_inv_shoup,
    const uint64_t* __restrict__ cnst,
    const uint64_t* __restrict__ cnst_shoup,
    const uint64_t* __restrict__ post_c0,
    const uint64_t* __restrict__ post_c1,
    const uint64_t* __restrict__ post_scalar,
    const uint64_t* __restrict__ primes,
    const int curr_limbs,
    const int sizeP,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup) {
  constexpr size_t LOG_RADIX = LOG_N - 6;
  constexpr int DATA_SIZE = 1u << LOG_RADIX;
  constexpr int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);
  constexpr int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;
  constexpr int N = 1 << LOG_N;

  const uint32_t limb = blockIdx.y;
  const uint32_t cv = blockIdx.z;
  const int end_length = curr_limbs - 1;
  power_of_roots += limb * N;
  power_of_roots_shoup += limb * N;
  const uint64_t prime = primes[limb];
  const uint64_t two_p = prime << 1;
  __shared__ uint64_t tile[2][NUM_WARP][WARP_SIZE + 1];

  uint64_t* out_cv = out + cv * end_length * N;
  const auto switched = hmul_ntt_phase2_pair<LOG_N, NUM_WARP>(
      out_cv,
      limb,
      blockIdx.x,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p,
      tile);

  const uint64_t switched_values[2] = {switched.x, switched.y};
  const int src_base = blockIdx.x * DATA_SIZE + 2 * threadIdx.x;
  const int in_LN = curr_limbs * N;
  const int inner_LN = (curr_limbs + sizeP) * N;
  const int out_LN = end_length * N;

#pragma unroll
  for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
    const int src = src_base + pair_idx;
    const int in_idx = limb * N + src;
    const int out_idx = cv * out_LN + in_idx;

    uint64_t key = sub_mod(
        inner_product[cv * inner_LN + in_idx],
        moddown_base[cv * inner_LN + in_idx],
        prime);
    key = mul_and_reduce_shoup(key, prod_inv[limb], prod_inv_shoup[limb], prime);
    if (key >= prime) {
      key -= prime;
    }

    uint64_t value = add_mod(raw[cv * in_LN + in_idx], key, prime);
    if constexpr (APPLY_DOUBLE) {
      value = add_mod(value, value, prime);
    }

    uint64_t scaled =
        mul_and_reduce_shoup(value, cnst[limb], cnst_shoup[limb], prime);
    if (scaled >= prime) {
      scaled -= prime;
    }

    uint64_t result = add_mod(switched_values[pair_idx], scaled, prime);
    if constexpr (POST_OP == HMulPostOp::AddCipher) {
      const uint64_t* post = (cv == 0) ? post_c0 : post_c1;
      result = add_mod(result, post[in_idx], prime);
    } else if constexpr (POST_OP == HMulPostOp::SubCipher) {
      const uint64_t* post = (cv == 0) ? post_c0 : post_c1;
      result = sub_mod(result, post[in_idx], prime);
    } else if constexpr (POST_OP == HMulPostOp::AddScalar) {
      if (cv == 0) {
        result = add_mod(result, post_scalar[limb], prime);
      }
    } else if constexpr (POST_OP == HMulPostOp::AddPlain) {
      if (cv == 0) {
        result = add_mod(result, post_c0[in_idx], prime);
      }
    }
    out[out_idx] = result;
  }
}

__global__ void hmul_innerproduct_without_original_copy_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ in_modup,
    const uint64_t* __restrict__ raw_axax,
    const uint64_t* __restrict__ eval_bx,
    const uint64_t* __restrict__ eval_ax,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barret_ratios,
    const uint64_t* __restrict__ barret_ks,
    const int N,
    const int length,
    const int mult_length,
    const int beta,
    const int curr_limbs,
    const int alpha,
    const int prime_gap,
    const int special_mod_start) {
  const int coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int limb = blockIdx.y;
  const int i = limb * N + coeff;
  const int swk_gap = special_mod_start - curr_limbs;
  const int prime_idx = (limb < curr_limbs) ? 0 : prime_gap;
  const int swk_idx = (limb < curr_limbs) ? 0 : swk_gap;
  const int reduce_prime_idx = limb + prime_idx;
  const uint64_t prime = primes[reduce_prime_idx];
  const uint64_t barret_ratio = barret_ratios[reduce_prime_idx];
  const uint64_t barret_k = barret_ks[reduce_prime_idx];
  const int original_beta = (limb < curr_limbs) ? (limb / alpha) : -1;
  const int in_stride = N * length;
  const int swk_stride = N * mult_length;

  uint128_t accum_ax{0};
  uint128_t accum_bx{0};
  int in_off = i;
  int swk_off = i + swk_idx * N;
  for (int beta_idx = 0; beta_idx < beta; ++beta_idx) {
    const uint64_t op1 =
        (beta_idx == original_beta) ? raw_axax[i] : in_modup[in_off];
    const auto mul_ax = mult_64_64_128(op1, eval_ax[swk_off]);
    const auto mul_bx = mult_64_64_128(op1, eval_bx[swk_off]);
    inplace_add_128_128(mul_ax, accum_ax);
    inplace_add_128_128(mul_bx, accum_bx);
    in_off += in_stride;
    swk_off += swk_stride;
  }

  out[i] = barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
  out[length * N + i] =
      barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
}

} // namespace fhe

namespace at::native {

static Tensor hmul_workspace_view(
    const Tensor& workspace,
    int64_t storage_offset,
    c10::IntArrayRef sizes) {
  int64_t stride = 1;
  std::vector<int64_t> strides(sizes.size());
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  TORCH_CHECK(
      storage_offset >= 0 && storage_offset + stride <= workspace.numel(),
      "hmul inner_workspace is too small: need ",
      storage_offset + stride,
      " uint64 values, got ",
      workspace.numel());
  return workspace.as_strided(sizes, strides, storage_offset);
}

template <bool APPLY_DOUBLE>
static void launch_hmul_relin_scale_last_limb(
    uint64_t* out_last,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const uint64_t* prod_inv,
    const uint64_t* prod_inv_shoup,
    const uint64_t* primes,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t N,
    cudaStream_t stream) {
  fhe::hmul_relin_scale_last_limb_kernel<APPLY_DOUBLE>
      <<<dim3(num_blocks(N), 2), BLOCK_SIZE, 0, stream>>>(
          out_last,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          primes,
          static_cast<int>(curr_limbs),
          static_cast<int>(sizeP),
          static_cast<int>(N));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void hmul_innerproduct_without_original_copy(
    const Tensor& out,
    const Tensor& modup,
    const Tensor& raw,
    const Tensor& swk_bx,
    const Tensor& swk_ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  const int64_t beta = (curr_limbs + alpha - 1) / alpha;
  const int64_t length = curr_limbs + sizeP;
  const int64_t mult_length = special_mod_start + sizeP;
  const int64_t prime_gap = L - curr_limbs;

  TORCH_CHECK(out.dim() == 4, "hmul inner_product workspace must be 4D");
  TORCH_CHECK(out.size(0) == 2 && out.size(1) == 1, "hmul inner_product workspace shape mismatch");
  TORCH_CHECK(out.size(2) == length && out.size(3) == N, "hmul inner_product workspace shape mismatch");
  TORCH_CHECK(out.is_contiguous(), "hmul inner_product workspace must be contiguous");
  const int64_t raw_LN = curr_limbs * N;
  const auto stream = at::cuda::getCurrentCUDAStream();
  fhe::hmul_innerproduct_without_original_copy_kernel<<<
      dim3(num_blocks(N), length),
      BLOCK_SIZE,
      0,
      stream>>>(
      out.data_ptr<uint64_t>(),
      modup.data_ptr<uint64_t>(),
      raw.data_ptr<uint64_t>() + 2 * raw_LN,
      swk_bx.data_ptr<uint64_t>(),
      swk_ax.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      static_cast<int>(N),
      static_cast<int>(length),
      static_cast<int>(mult_length),
      static_cast<int>(beta),
      static_cast<int>(curr_limbs),
      static_cast<int>(alpha),
      static_cast<int>(prime_gap),
      static_cast<int>(special_mod_start));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N>
static void launch_hmul_moddown_base_convert_ntt_phase1(
    uint64_t* workspace_ptr,
    int64_t L_IN,
    int64_t curr_limbs,
    int64_t sizeP,
    const uint64_t* prod_q_i_mod_q_j_moddown,
    const uint64_t* primes,
    const uint64_t* barret_ratio,
    const uint64_t* barret_k,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t NUM_GROUPS = 8;
  fhe::hmul_moddown_base_convert_ntt_phase1_kernel<LOG_N, NUM_GROUPS>
      <<<dim3(N / (NUM_GROUPS * 8) / 8, curr_limbs, 2),
         NUM_GROUPS * 8,
         sizeP * sizeof(uint64_t),
         stream>>>(
          workspace_ptr,
          L_IN,
          curr_limbs,
          sizeP,
          prod_q_i_mod_q_j_moddown,
          primes,
          barret_ratio,
          barret_k,
          power_of_roots,
          power_of_roots_shoup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N>
static void launch_hmul_ntt_phase2_store(
    uint64_t* workspace_ptr,
    int64_t L_IN,
    int64_t curr_limbs,
    const uint64_t* primes,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;
  fhe::hmul_ntt_phase2_store_kernel<LOG_N>
      <<<dim3(RADIX, curr_limbs, 2), N / RADIX / 2, 0, stream>>>(
          workspace_ptr,
          L_IN,
          curr_limbs,
          primes,
          power_of_roots,
          power_of_roots_shoup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N>
static void launch_hmul_moddown_base_convert_ntt(
    uint64_t* workspace_ptr,
    int64_t L_IN,
    int64_t curr_limbs,
    int64_t sizeP,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup,
    cudaStream_t stream) {
  launch_hmul_moddown_base_convert_ntt_phase1<LOG_N>(
      workspace_ptr,
      L_IN,
      curr_limbs,
      sizeP,
      prod_q_i_mod_q_j_moddown.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      stream);
  launch_hmul_ntt_phase2_store<LOG_N>(
      workspace_ptr,
      L_IN,
      curr_limbs,
      primes.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      stream);
}

static void hmul_moddown_base_convert_ntt_cuda(
    uint64_t* workspace_ptr,
    int64_t L_IN,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t N,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (N == (int64_t{1} << 17)) {
    launch_hmul_moddown_base_convert_ntt<17>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown,
        primes,
        barret_ratio,
        barret_k,
        power_of_roots,
        power_of_roots_shoup,
        stream);
  } else if (N == (int64_t{1} << 16)) {
    launch_hmul_moddown_base_convert_ntt<16>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown,
        primes,
        barret_ratio,
        barret_k,
        power_of_roots,
        power_of_roots_shoup,
        stream);
  } else if (N == (int64_t{1} << 15)) {
    launch_hmul_moddown_base_convert_ntt<15>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown,
        primes,
        barret_ratio,
        barret_k,
        power_of_roots,
        power_of_roots_shoup,
        stream);
  } else if (N == (int64_t{1} << 14)) {
    launch_hmul_moddown_base_convert_ntt<14>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown,
        primes,
        barret_ratio,
        barret_k,
        power_of_roots,
        power_of_roots_shoup,
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported hmul NTT size");
  }
}

template <size_t LOG_N>
static void launch_hmul_switch_const_mult_ntt_phase1(
    Tensor& out,
    const uint64_t* last_limb_coeff_ptr,
    const uint64_t* cnst,
    const uint64_t* cnst_shoup,
    const uint64_t* primes,
    const uint64_t* diffs,
    uint64_t old_modulus_by_two,
    int64_t curr_limbs,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t NUM_GROUPS = 8;
  fhe::hmul_switch_const_mult_ntt_phase1_kernel<LOG_N, NUM_GROUPS>
      <<<dim3(N / (NUM_GROUPS * 8) / 8, curr_limbs - 1, 2),
         NUM_GROUPS * 8,
         0,
         stream>>>(
          out.data_ptr<uint64_t>(),
          last_limb_coeff_ptr,
          cnst,
          cnst_shoup,
          primes,
          diffs,
          old_modulus_by_two,
          static_cast<int>(curr_limbs),
          power_of_roots,
          power_of_roots_shoup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N, bool APPLY_DOUBLE, fhe::HMulPostOp POST_OP>
static void launch_hmul_ntt_phase2_finalize(
    Tensor& out,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const uint64_t* prod_inv,
    const uint64_t* prod_inv_shoup,
    const uint64_t* cnst,
    const uint64_t* cnst_shoup,
    const uint64_t* post_c0,
    const uint64_t* post_c1,
    const uint64_t* post_scalar,
    const uint64_t* primes,
    int64_t curr_limbs,
    int64_t sizeP,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;
  fhe::hmul_ntt_phase2_finalize_kernel<LOG_N, APPLY_DOUBLE, POST_OP>
      <<<dim3(RADIX, curr_limbs - 1, 2), N / RADIX / 2, 0, stream>>>(
          out.data_ptr<uint64_t>(),
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          static_cast<int>(curr_limbs),
          static_cast<int>(sizeP),
          power_of_roots,
          power_of_roots_shoup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N, bool APPLY_DOUBLE>
static void dispatch_hmul_ntt_phase2_finalize(
    int64_t post_op,
    Tensor& out,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const uint64_t* prod_inv,
    const uint64_t* prod_inv_shoup,
    const uint64_t* cnst,
    const uint64_t* cnst_shoup,
    const uint64_t* post_c0,
    const uint64_t* post_c1,
    const uint64_t* post_scalar,
    const uint64_t* primes,
    int64_t curr_limbs,
    int64_t sizeP,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup,
    cudaStream_t stream) {
  switch (post_op) {
    case static_cast<int64_t>(fhe::HMulPostOp::None):
      launch_hmul_ntt_phase2_finalize<LOG_N, APPLY_DOUBLE, fhe::HMulPostOp::None>(
          out,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          power_of_roots,
          power_of_roots_shoup,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::AddCipher):
      launch_hmul_ntt_phase2_finalize<LOG_N, APPLY_DOUBLE, fhe::HMulPostOp::AddCipher>(
          out,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          power_of_roots,
          power_of_roots_shoup,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::SubCipher):
      launch_hmul_ntt_phase2_finalize<LOG_N, APPLY_DOUBLE, fhe::HMulPostOp::SubCipher>(
          out,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          power_of_roots,
          power_of_roots_shoup,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::AddScalar):
      launch_hmul_ntt_phase2_finalize<LOG_N, APPLY_DOUBLE, fhe::HMulPostOp::AddScalar>(
          out,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          power_of_roots,
          power_of_roots_shoup,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::AddPlain):
      launch_hmul_ntt_phase2_finalize<LOG_N, APPLY_DOUBLE, fhe::HMulPostOp::AddPlain>(
          out,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          power_of_roots,
          power_of_roots_shoup,
          stream);
      return;
    default:
      TORCH_CHECK(false, "unsupported hmul post_op: ", post_op);
  }
}

template <size_t LOG_N, bool APPLY_DOUBLE>
static void launch_hmul_switch_const_mult_ntt_finalize(
    Tensor& out,
    const uint64_t* last_limb_coeff_ptr,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const uint64_t* prod_inv,
    const uint64_t* prod_inv_shoup,
    const uint64_t* switch_cnst,
    const uint64_t* switch_cnst_shoup,
    const uint64_t* final_cnst,
    const uint64_t* final_cnst_shoup,
    const uint64_t* post_c0,
    const uint64_t* post_c1,
    const uint64_t* post_scalar,
    const uint64_t* primes,
    const uint64_t* diffs,
    uint64_t old_modulus_by_two,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t post_op,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup,
    cudaStream_t stream) {
  launch_hmul_switch_const_mult_ntt_phase1<LOG_N>(
      out,
      last_limb_coeff_ptr,
      switch_cnst,
      switch_cnst_shoup,
      primes,
      diffs,
      old_modulus_by_two,
      curr_limbs,
      power_of_roots,
      power_of_roots_shoup,
      stream);
  dispatch_hmul_ntt_phase2_finalize<LOG_N, APPLY_DOUBLE>(
      post_op,
      out,
      raw,
      inner_ptr,
      moddown_base_ptr,
      prod_inv,
      prod_inv_shoup,
      final_cnst,
      final_cnst_shoup,
      post_c0,
      post_c1,
      post_scalar,
      primes,
      curr_limbs,
      sizeP,
      power_of_roots,
      power_of_roots_shoup,
      stream);
}

template <bool APPLY_DOUBLE>
static void hmul_switch_const_mult_ntt_finalize_cuda(
    Tensor& out,
    const uint64_t* last_limb_coeff_ptr,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const uint64_t* switch_cnst,
    const uint64_t* switch_cnst_shoup,
    const uint64_t* final_cnst,
    const uint64_t* final_cnst_shoup,
    const uint64_t* post_c0,
    const uint64_t* post_c1,
    const uint64_t* post_scalar,
    const Tensor& primes,
    const uint64_t* diffs,
    uint64_t old_modulus_by_two,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t N,
    int64_t post_op,
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (N == (int64_t{1} << 17)) {
    launch_hmul_switch_const_mult_ntt_finalize<17, APPLY_DOUBLE>(
        out,
        last_limb_coeff_ptr,
        raw,
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        switch_cnst,
        switch_cnst_shoup,
        final_cnst,
        final_cnst_shoup,
        post_c0,
        post_c1,
        post_scalar,
        primes.data_ptr<uint64_t>(),
        diffs,
        old_modulus_by_two,
        curr_limbs,
        sizeP,
        post_op,
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else if (N == (int64_t{1} << 16)) {
    launch_hmul_switch_const_mult_ntt_finalize<16, APPLY_DOUBLE>(
        out,
        last_limb_coeff_ptr,
        raw,
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        switch_cnst,
        switch_cnst_shoup,
        final_cnst,
        final_cnst_shoup,
        post_c0,
        post_c1,
        post_scalar,
        primes.data_ptr<uint64_t>(),
        diffs,
        old_modulus_by_two,
        curr_limbs,
        sizeP,
        post_op,
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else if (N == (int64_t{1} << 15)) {
    launch_hmul_switch_const_mult_ntt_finalize<15, APPLY_DOUBLE>(
        out,
        last_limb_coeff_ptr,
        raw,
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        switch_cnst,
        switch_cnst_shoup,
        final_cnst,
        final_cnst_shoup,
        post_c0,
        post_c1,
        post_scalar,
        primes.data_ptr<uint64_t>(),
        diffs,
        old_modulus_by_two,
        curr_limbs,
        sizeP,
        post_op,
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else if (N == (int64_t{1} << 14)) {
    launch_hmul_switch_const_mult_ntt_finalize<14, APPLY_DOUBLE>(
        out,
        last_limb_coeff_ptr,
        raw,
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        switch_cnst,
        switch_cnst_shoup,
        final_cnst,
        final_cnst_shoup,
        post_c0,
        post_c1,
        post_scalar,
        primes.data_ptr<uint64_t>(),
        diffs,
        old_modulus_by_two,
        curr_limbs,
        sizeP,
        post_op,
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported hmul NTT size");
  }
}

static Tensor hmul_moddown_drop_last_scale(
    const Tensor& raw,
    const Tensor& inner_product,
    const Tensor& moddown_workspace,
    const Tensor& workspace,
    const Tensor& last_limb_ntt,
    const std::optional<Tensor>& post_c0,
    const std::optional<Tensor>& post_c1,
    const std::optional<Tensor>& post_scalar,
    int64_t curr_limbs,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    int64_t old_prime,
    bool apply_double,
    int64_t post_op,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& switch_modulus_map,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  constexpr int64_t num_cv = 2;
  constexpr int64_t num_cipher = 1;
  const int64_t end_length = curr_limbs - 1;
  if (post_op == static_cast<int64_t>(fhe::HMulPostOp::AddCipher) ||
      post_op == static_cast<int64_t>(fhe::HMulPostOp::SubCipher)) {
    TORCH_CHECK(post_c0.has_value(), "hmul post cipher op requires post_c0/post_c1");
    TORCH_CHECK(!post_scalar.has_value(), "hmul post cipher op cannot also use post_scalar");
    TORCH_CHECK(post_c0->is_contiguous(), "hmul post_c0 must be contiguous");
    TORCH_CHECK(post_c1->is_contiguous(), "hmul post_c1 must be contiguous");
    TORCH_CHECK(post_c0->dim() == 2 && post_c1->dim() == 2, "hmul post cipher must be [limbs, N]");
    TORCH_CHECK(post_c0->sizes()[0] >= end_length && post_c0->sizes()[1] == N, "hmul post_c0 shape mismatch");
    TORCH_CHECK(post_c1->sizes()[0] >= end_length && post_c1->sizes()[1] == N, "hmul post_c1 shape mismatch");
  } else if (post_op == static_cast<int64_t>(fhe::HMulPostOp::AddScalar)) {
    TORCH_CHECK(post_scalar.has_value(), "hmul add scalar op requires post_scalar");
    TORCH_CHECK(!post_c0.has_value() && !post_c1.has_value(), "hmul scalar op cannot also use post cipher");
    TORCH_CHECK(post_scalar->is_contiguous(), "hmul post_scalar must be contiguous");
    TORCH_CHECK(post_scalar->dim() == 1 && post_scalar->sizes()[0] == end_length, "hmul post_scalar shape mismatch");
  } else if (post_op == static_cast<int64_t>(fhe::HMulPostOp::AddPlain)) {
    TORCH_CHECK(post_c0.has_value(), "hmul add plaintext op requires post_c0");
    TORCH_CHECK(!post_c1.has_value(), "hmul add plaintext op cannot use post_c1");
    TORCH_CHECK(!post_scalar.has_value(), "hmul add plaintext op cannot also use post_scalar");
    TORCH_CHECK(post_c0->is_contiguous(), "hmul post plaintext must be contiguous");
    TORCH_CHECK(post_c0->dim() == 2, "hmul post plaintext must be [limbs, N]");
    TORCH_CHECK(post_c0->sizes()[0] >= end_length && post_c0->sizes()[1] == N, "hmul post plaintext shape mismatch");
  } else {
    TORCH_CHECK(!post_c0.has_value() && !post_c1.has_value(), "hmul no-post op got post cipher");
    TORCH_CHECK(!post_scalar.has_value(), "hmul no-post op got post scalar");
  }

  auto out = at::empty({num_cv, num_cipher, end_length, N}, raw.options());
  TORCH_CHECK(
      moddown_workspace.sizes() ==
          c10::IntArrayRef({num_cv, num_cipher, curr_limbs + sizeP, N}),
      "hmul moddown_workspace shape mismatch");
  TORCH_CHECK(
      workspace.sizes() == c10::IntArrayRef({num_cv, num_cipher, curr_limbs, N}),
      "hmul workspace shape mismatch");
  TORCH_CHECK(
      last_limb_ntt.sizes() == c10::IntArrayRef({num_cv, num_cipher, 1, N}),
      "hmul last_limb_ntt workspace shape mismatch");
  TORCH_CHECK(moddown_workspace.is_contiguous(), "hmul moddown_workspace must be contiguous");
  TORCH_CHECK(workspace.is_contiguous(), "hmul workspace must be contiguous");
  TORCH_CHECK(last_limb_ntt.is_contiguous(), "hmul last_limb_ntt must be contiguous");

  const auto stream = at::cuda::getCurrentCUDAStream();

  auto* inner_ptr = inner_product.data_ptr<uint64_t>();
  auto* moddown_workspace_ptr = moddown_workspace.data_ptr<uint64_t>();
  auto* moddown_base_ptr = moddown_workspace_ptr;

  iNTT_scaled_impl(
      moddown_workspace_ptr + curr_limbs * N,
      inner_ptr + curr_limbs * N,
      sizeP,
      N,
      curr_limbs + sizeP,
      curr_limbs + sizeP,
      num_cv,
      num_cipher,
      primes.data_ptr<uint64_t>() + L,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + L * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() + L * N,
      hat_inverse_vec_moddown.data_ptr<uint64_t>(),
      hat_inverse_vec_shoup_moddown.data_ptr<uint64_t>());

  hmul_moddown_base_convert_ntt_cuda(
      moddown_base_ptr,
      curr_limbs + sizeP,
      curr_limbs,
      sizeP,
      N,
      prod_q_i_mod_q_j_moddown,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots,
      power_of_roots_shoup);

  if (apply_double) {
    launch_hmul_relin_scale_last_limb<true>(
        last_limb_ntt.data_ptr<uint64_t>(),
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        curr_limbs,
        sizeP,
        N,
        stream);
  } else {
    launch_hmul_relin_scale_last_limb<false>(
        last_limb_ntt.data_ptr<uint64_t>(),
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        curr_limbs,
        sizeP,
        N,
        stream);
  }

  auto* workspace_ptr = workspace.data_ptr<uint64_t>();
  iNTT_impl(
      workspace_ptr + N * end_length,
      last_limb_ntt.data_ptr<uint64_t>(),
      1,
      N,
      curr_limbs,
      1,
      num_cv,
      num_cipher,
      primes.data_ptr<uint64_t>() + end_length,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + end_length * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() +
          end_length * N);

  const int64_t switch_op2_idx = (L - curr_limbs) * (L - 1);
  const int64_t final_op2_idx = end_length * L;
  const uint64_t* post_c0_ptr =
      post_c0.has_value() ? post_c0->data_ptr<uint64_t>() : nullptr;
  const uint64_t* post_c1_ptr =
      post_c1.has_value() ? post_c1->data_ptr<uint64_t>() : nullptr;
  const uint64_t* post_scalar_ptr =
      post_scalar.has_value() ? post_scalar->data_ptr<uint64_t>() : nullptr;
  if (apply_double) {
    hmul_switch_const_mult_ntt_finalize_cuda<true>(
        out,
        workspace_ptr + N * end_length,
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown,
        prod_inv_shoup_moddown,
        qlql_inv_mod_ql_div_ql_mod_q.data_ptr<uint64_t>() + switch_op2_idx,
        qlql_inv_mod_ql_div_ql_mod_q_shoup.data_ptr<uint64_t>() + switch_op2_idx,
        q_inv_mod_q.data_ptr<uint64_t>() + final_op2_idx,
        q_inv_mod_q_shoup.data_ptr<uint64_t>() + final_op2_idx,
        post_c0_ptr,
        post_c1_ptr,
        post_scalar_ptr,
        primes,
        switch_modulus_map.data_ptr<uint64_t>() + end_length * primes.numel(),
        static_cast<uint64_t>(old_prime) >> 1,
        curr_limbs,
        sizeP,
        N,
        post_op,
        power_of_roots,
        power_of_roots_shoup);
  } else {
    hmul_switch_const_mult_ntt_finalize_cuda<false>(
        out,
        workspace_ptr + N * end_length,
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown,
        prod_inv_shoup_moddown,
        qlql_inv_mod_ql_div_ql_mod_q.data_ptr<uint64_t>() + switch_op2_idx,
        qlql_inv_mod_ql_div_ql_mod_q_shoup.data_ptr<uint64_t>() + switch_op2_idx,
        q_inv_mod_q.data_ptr<uint64_t>() + final_op2_idx,
        q_inv_mod_q_shoup.data_ptr<uint64_t>() + final_op2_idx,
        post_c0_ptr,
        post_c1_ptr,
        post_scalar_ptr,
        primes,
        switch_modulus_map.data_ptr<uint64_t>() + end_length * primes.numel(),
        static_cast<uint64_t>(old_prime) >> 1,
        curr_limbs,
        sizeP,
        N,
        post_op,
        power_of_roots,
        power_of_roots_shoup);
  }

  return out;
}

static std::vector<Tensor> hmul_double_rescale_impl(
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& d0,
    const Tensor& d1,
    const Tensor& swk_bx,
    const Tensor& swk_ax,
    const std::optional<Tensor>& post_c0,
    const std::optional<Tensor>& post_c1,
    const std::optional<Tensor>& post_scalar,
    int64_t curr_limbs,
    int64_t special_mod_start,
    int64_t L,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    int64_t old_prime,
    const Tensor& primes,
    const Tensor& q_mu,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& hat_inverse_vec_modup,
    const Tensor& hat_inverse_vec_shoup_modup,
    const Tensor& prod_q_i_mod_q_j_modup,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& switch_modulus_map,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    const Tensor& inner_workspace,
    bool apply_double,
    int64_t post_op) {
  TORCH_CHECK(c0.is_contiguous(), "hmul c0 must be contiguous");
  TORCH_CHECK(c1.is_contiguous(), "hmul c1 must be contiguous");
  TORCH_CHECK(d0.is_contiguous(), "hmul d0 must be contiguous");
  TORCH_CHECK(d1.is_contiguous(), "hmul d1 must be contiguous");
  TORCH_CHECK(c0.dim() == 3 && c0.size(0) == 1, "hmul inputs must be [1, limbs, N]");
  TORCH_CHECK(c1.dim() == 3 && c1.size(0) == 1, "hmul inputs must be [1, limbs, N]");
  TORCH_CHECK(c0.size(1) >= curr_limbs && c0.size(2) == N, "hmul c0 shape mismatch");
  TORCH_CHECK(c1.size(1) >= curr_limbs && c1.size(2) == N, "hmul c1 shape mismatch");
  TORCH_CHECK(d0.size(1) >= curr_limbs && d0.size(2) == N, "hmul d0 shape mismatch");
  TORCH_CHECK(d1.size(1) >= curr_limbs && d1.size(2) == N, "hmul d1 shape mismatch");
  TORCH_CHECK(inner_workspace.is_contiguous(), "hmul inner_workspace must be contiguous");
  TORCH_CHECK(inner_workspace.is_cuda() == c0.is_cuda(), "hmul inner_workspace device mismatch");
  TORCH_CHECK(alpha > 0, "hmul alpha must be positive");
  TORCH_CHECK(curr_limbs > 1, "hmul curr_limbs must be greater than 1");
  TORCH_CHECK(L >= curr_limbs, "hmul L must be >= curr_limbs");
  TORCH_CHECK(
      beta == (curr_limbs + alpha - 1) / alpha,
      "hmul beta mismatch");
  TORCH_CHECK(
      special_mod_start >= curr_limbs,
      "hmul special_mod_start must be >= curr_limbs");
  TORCH_CHECK(
      special_mod_start <= L,
      "hmul special_mod_start must be <= L");
  TORCH_CHECK(
      N == (int64_t{1} << 14) || N == (int64_t{1} << 15) ||
          N == (int64_t{1} << 16) || N == (int64_t{1} << 17),
      "hmul unsupported N: ",
      N);
  TORCH_CHECK(primes.dim() == 1, "hmul primes must be 1D");
  TORCH_CHECK(barret_ratio.dim() == 1, "hmul barret_ratio must be 1D");
  TORCH_CHECK(barret_k.dim() == 1, "hmul barret_k must be 1D");
  TORCH_CHECK(primes.is_contiguous(), "hmul primes must be contiguous");
  TORCH_CHECK(q_mu.is_contiguous(), "hmul q_mu must be contiguous");
  TORCH_CHECK(barret_ratio.is_contiguous(), "hmul barret_ratio must be contiguous");
  TORCH_CHECK(barret_k.is_contiguous(), "hmul barret_k must be contiguous");
  TORCH_CHECK(primes.numel() > L, "hmul requires special P limbs");
  TORCH_CHECK(q_mu.numel() >= 2 * L, "hmul q_mu shape mismatch");
  TORCH_CHECK(barret_ratio.numel() >= L, "hmul barret_ratio shape mismatch");
  TORCH_CHECK(barret_k.numel() >= L, "hmul barret_k shape mismatch");
  TORCH_CHECK(swk_bx.is_contiguous(), "hmul swk_bx must be contiguous");
  TORCH_CHECK(swk_ax.is_contiguous(), "hmul swk_ax must be contiguous");
  TORCH_CHECK(swk_bx.dim() == 3 && swk_ax.dim() == 3, "hmul swk must be [beta, limbs, N]");
  TORCH_CHECK(swk_bx.sizes() == swk_ax.sizes(), "hmul swk shape mismatch");

  const int64_t sizeP = primes.numel() - L;
  TORCH_CHECK(sizeP > 0, "hmul sizeP must be positive");
  TORCH_CHECK(
      swk_bx.size(0) >= beta,
      "hmul swk beta dimension mismatch");
  TORCH_CHECK(
      swk_bx.size(1) >= special_mod_start + sizeP,
      "hmul swk modulus dimension mismatch");
  TORCH_CHECK(swk_bx.size(2) == N, "hmul swk N dimension mismatch");
  TORCH_CHECK(N <= std::numeric_limits<int>::max(), "hmul N exceeds int range");
  TORCH_CHECK(
      curr_limbs + sizeP <= std::numeric_limits<int>::max(),
      "hmul limb count exceeds int range");
  TORCH_CHECK(
      curr_limbs * N <= std::numeric_limits<int>::max(),
      "hmul Q workspace exceeds int index range");
  TORCH_CHECK(
      (curr_limbs + sizeP) * N <= std::numeric_limits<int>::max(),
      "hmul QP workspace exceeds int index range");
  TORCH_CHECK(
      beta * (curr_limbs + sizeP) * N <= std::numeric_limits<int>::max(),
      "hmul modup workspace exceeds int index range");
  TORCH_CHECK(
      (special_mod_start + sizeP) * N <= std::numeric_limits<int>::max(),
      "hmul switching key stride exceeds int index range");
  const int64_t length = curr_limbs + sizeP;
  int64_t workspace_offset = 0;
  auto raw = hmul_workspace_view(
      inner_workspace,
      workspace_offset,
      {3, 1, curr_limbs, N});
  workspace_offset += raw.numel();
  auto modup = hmul_workspace_view(
      inner_workspace,
      workspace_offset,
      {1, beta * length, N});
  workspace_offset += modup.numel();
  auto modup_temp = hmul_workspace_view(
      inner_workspace,
      workspace_offset,
      {1, curr_limbs, N});
  workspace_offset += modup_temp.numel();
  auto inner_product = hmul_workspace_view(
      inner_workspace,
      workspace_offset,
      {2, 1, length, N});
  workspace_offset += inner_product.numel();
  auto moddown_workspace = hmul_workspace_view(
      inner_workspace,
      workspace_offset,
      {2, 1, length, N});
  workspace_offset += moddown_workspace.numel();
  auto workspace = hmul_workspace_view(
      inner_workspace,
      workspace_offset,
      {2, 1, curr_limbs, N});
  workspace_offset += workspace.numel();
  auto last_limb_ntt = hmul_workspace_view(
      inner_workspace,
      workspace_offset,
      {2, 1, 1, N});

  const dim3 block(BLOCK_SIZE);
  const dim3 grid(num_blocks(N), curr_limbs);
  const auto stream = at::cuda::getCurrentCUDAStream();

  fhe::hmul_raw_kernel<<<grid, block, 0, stream>>>(
      raw.data_ptr<uint64_t>(),
      c0.data_ptr<uint64_t>(),
      c1.data_ptr<uint64_t>(),
      d0.data_ptr<uint64_t>(),
      d1.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      q_mu.data_ptr<uint64_t>(),
      static_cast<int>(curr_limbs),
      static_cast<int>(N));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto axax = raw.slice(0, 2, 3).reshape({1, curr_limbs, N});
  modup_without_copy_cuda_out(
      modup,
      modup_temp,
      axax,
      curr_limbs,
      L,
      beta,
      N,
      alpha,
      hat_inverse_vec_modup,
      hat_inverse_vec_shoup_modup,
      prod_q_i_mod_q_j_modup,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two);

  hmul_innerproduct_without_original_copy(
      inner_product,
      modup,
      raw,
      swk_bx,
      swk_ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      sizeP,
      N,
      primes,
      barret_ratio,
      barret_k);

  std::optional<Tensor> post_c0_2d = post_c0;
  std::optional<Tensor> post_c1_2d = post_c1;
  if (post_c0_2d.has_value() && post_c0_2d->dim() == 3) {
    post_c0_2d = (*post_c0_2d)[0];
  }
  if (post_c1_2d.has_value() && post_c1_2d->dim() == 3) {
    post_c1_2d = (*post_c1_2d)[0];
  }

  auto result = hmul_moddown_drop_last_scale(
      raw,
      inner_product,
      moddown_workspace,
      workspace,
      last_limb_ntt,
      post_c0_2d,
      post_c1_2d,
      post_scalar,
      curr_limbs,
      L,
      sizeP,
      N,
      old_prime,
      apply_double,
      post_op,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      primes,
      barret_ratio,
      barret_k,
      switch_modulus_map,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      q_inv_mod_q,
      q_inv_mod_q_shoup);
  return {result[0], result[1]};
}

std::vector<Tensor> hmul_double_rescale_cuda(
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& d0,
    const Tensor& d1,
    const Tensor& swk_bx,
    const Tensor& swk_ax,
    int64_t curr_limbs,
    int64_t special_mod_start,
    int64_t L,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    int64_t old_prime,
    const Tensor& primes,
    const Tensor& q_mu,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& hat_inverse_vec_modup,
    const Tensor& hat_inverse_vec_shoup_modup,
    const Tensor& prod_q_i_mod_q_j_modup,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& switch_modulus_map,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    const Tensor& inner_workspace,
    bool apply_double,
    int64_t post_op,
    const std::optional<Tensor>& post_c0,
    const std::optional<Tensor>& post_c1,
    const std::optional<Tensor>& post_scalar) {
  return hmul_double_rescale_impl(
      c0,
      c1,
      d0,
      d1,
      swk_bx,
      swk_ax,
      post_c0,
      post_c1,
      post_scalar,
      curr_limbs,
      special_mod_start,
      L,
      beta,
      N,
      alpha,
      old_prime,
      primes,
      q_mu,
      barret_ratio,
      barret_k,
      hat_inverse_vec_modup,
      hat_inverse_vec_shoup_modup,
      prod_q_i_mod_q_j_modup,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      switch_modulus_map,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      inner_workspace,
      apply_double,
      post_op);
}

} // namespace at::native
