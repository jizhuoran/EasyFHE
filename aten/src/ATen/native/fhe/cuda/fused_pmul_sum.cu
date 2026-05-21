#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__global__ void fusedPairwiseMACKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mods,
    const uint64_t* __restrict__ barret_ratio,
    const uint64_t* __restrict__ barret_k,
    const int64_t num_ciphers,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN,
    const int64_t L_PTN) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto bid_y = blockIdx.y;

  uint128_t sum_bx = {0, 0};
  uint128_t sum_ax = {0, 0};

  for (size_t i = 0; i < num_ciphers; ++i) {
    auto plain_val = plain_ptr[i * L_PTN + bid_y * N + tid_x];
    auto cipher_off = i * L_CTN + bid_y * N + tid_x;
    auto cipher_val_bx = cipher_ptr[cipher_off];
    auto cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
    inplace_add_128_128(mult_64_64_128(cipher_val_bx, plain_val), sum_bx);
    inplace_add_128_128(mult_64_64_128(cipher_val_ax, plain_val), sum_ax);
  }

  auto mod = mods[bid_y];
  res_ptr[bid_y * N + tid_x] = barret_reduction_128_64(
      sum_bx, mod, barret_ratio[bid_y], barret_k[bid_y]);
  res_ptr[cur_limbs * N + bid_y * N + tid_x] = barret_reduction_128_64(
      sum_ax, mod, barret_ratio[bid_y], barret_k[bid_y]);
}

__global__ void fusedPairwiseMACKernel_batch(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t num_batches,
    const int64_t num_ciphers,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN,
    const int64_t L_PTN) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto bid_y = blockIdx.y;

  extern __shared__ uint64_t cipher_shared[];

  auto mod = mod_ptr[bid_y];
  auto barret_ratio = barret_ratio_ptr[bid_y];
  auto barret_k = barret_k_ptr[bid_y];

  uint128_t sum_bx = {0, 0};
  uint128_t sum_ax = {0, 0};
  for (size_t i = 0; i < num_ciphers; ++i) {
    auto plain_val = plain_ptr[i * L_PTN + bid_y * N + tid_x];
    auto cipher_off = i * L_CTN + bid_y * N + tid_x;
    auto cipher_val_bx = cipher_ptr[cipher_off];
    auto cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
    inplace_add_128_128(mult_64_64_128(cipher_val_bx, plain_val), sum_bx);
    inplace_add_128_128(mult_64_64_128(cipher_val_ax, plain_val), sum_ax);
    cipher_shared[BLOCK_SIZE * i + threadIdx.x] = cipher_val_bx;
    cipher_shared[BLOCK_SIZE * (num_ciphers + i) + threadIdx.x] = cipher_val_ax;
  }
  res_ptr[bid_y * N + tid_x] =
      barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
  res_ptr[num_batches * cur_limbs * N + bid_y * N + tid_x] =
      barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);

  __syncthreads();
  for (int batch_id = 1; batch_id < num_batches; ++batch_id) {
    uint128_t sum_bx = {0, 0};
    uint128_t sum_ax = {0, 0};
    for (size_t i = 0; i < num_ciphers; ++i) {
      auto plain_val =
          plain_ptr[(batch_id * num_ciphers + i) * L_PTN + bid_y * N + tid_x];
      auto cipher_off = i * BLOCK_SIZE + threadIdx.x;
      auto cipher_val_bx = cipher_shared[cipher_off];
      auto cipher_val_ax = cipher_shared[cipher_off + num_ciphers * BLOCK_SIZE];
      inplace_add_128_128(mult_64_64_128(cipher_val_bx, plain_val), sum_bx);
      inplace_add_128_128(mult_64_64_128(cipher_val_ax, plain_val), sum_ax);
    }
    res_ptr[batch_id * cur_limbs * N + bid_y * N + tid_x] =
        barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
    res_ptr
        [num_batches * cur_limbs * N + batch_id * cur_limbs * N + bid_y * N +
         tid_x] = barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);
  }
}

__global__ void cpmulBroadcastPTKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_mu_ptr,
    const int64_t num_ciphers,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN) {

  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto bid_y = blockIdx.y;

  auto mod = mod_ptr[bid_y];
  auto barret_mu0 = barret_mu_ptr[bid_y * 2];
  auto barret_mu1 = barret_mu_ptr[bid_y * 2 + 1];
  auto ptx_val = plain_ptr[bid_y * N + tid_x];


  for (int batch_id = 0; batch_id < num_ciphers; ++batch_id) {
    auto cipher_off = batch_id * L_CTN + bid_y * N + tid_x;
    auto cipher_val_bx = cipher_ptr[cipher_off];
    auto cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];

    res_ptr[cipher_off] = mul_mod(cipher_val_bx, ptx_val, mod, barret_mu0, barret_mu1);
    res_ptr[cipher_off + BL_CTN] =
        mul_mod(cipher_val_ax, ptx_val, mod, barret_mu0, barret_mu1);
  }
}

__global__ void fusedBroadcastMACKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t num_plain,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t L_PTN) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto bid_y = blockIdx.y;

  auto mod = mod_ptr[bid_y];
  auto barret_ratio = barret_ratio_ptr[bid_y];
  auto barret_k = barret_k_ptr[bid_y];
  auto cipher_val_bx = cipher_ptr[bid_y * N + tid_x];
  auto cipher_val_ax = cipher_ptr[L_CTN + bid_y * N + tid_x];

  uint128_t sum_bx = {0, 0};
  uint128_t sum_ax = {0, 0};
  for (int64_t i = 0; i < num_plain; ++i) {
    auto plain_val = plain_ptr[i * L_PTN + bid_y * N + tid_x];
    inplace_add_128_128(mult_64_64_128(cipher_val_bx, plain_val), sum_bx);
    inplace_add_128_128(mult_64_64_128(cipher_val_ax, plain_val), sum_ax);
  }

  res_ptr[bid_y * N + tid_x] =
      barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
  res_ptr[cur_limbs * N + bid_y * N + tid_x] =
      barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);
}

__global__ void scalarWeightedAccKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ scalar_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t num_cipher,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto bid_y = blockIdx.y;

  auto mod = mod_ptr[bid_y];
  auto barret_ratio = barret_ratio_ptr[bid_y];
  auto barret_k = barret_k_ptr[bid_y];

  uint128_t sum_bx = {0, 0};
  uint128_t sum_ax = {0, 0};
  for (int64_t i = 0; i < num_cipher; ++i) {
    auto scalar_val = scalar_ptr[i * cur_limbs + bid_y];
    auto cipher_off = i * L_CTN + bid_y * N + tid_x;
    auto cipher_val_bx = cipher_ptr[cipher_off];
    auto cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
    inplace_add_128_128(mult_64_64_128(cipher_val_bx, scalar_val), sum_bx);
    inplace_add_128_128(mult_64_64_128(cipher_val_ax, scalar_val), sum_ax);
  }

  res_ptr[bid_y * N + tid_x] =
      barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
  res_ptr[cur_limbs * N + bid_y * N + tid_x] =
      barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);
}

__global__ void groupedScalarWeightedAccGridKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ scalar_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t num_groups,
    const int64_t num_cipher,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN) {
  const int64_t tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (tid_x >= N) {
    return;
  }

  const int64_t limb = blockIdx.y;
  const int64_t group = blockIdx.z;
  if (group >= num_groups) {
    return;
  }

  const uint64_t mod = mod_ptr[limb];
  const uint64_t barret_ratio = barret_ratio_ptr[limb];
  const uint64_t barret_k = barret_k_ptr[limb];

  uint128_t sum_bx = {0, 0};
  uint128_t sum_ax = {0, 0};
  for (int64_t i = 0; i < num_cipher; ++i) {
    const uint64_t scalar_val =
        scalar_ptr[(group * num_cipher + i) * cur_limbs + limb];
    const int64_t cipher_off = i * L_CTN + limb * N + tid_x;
    const uint64_t cipher_val_bx = cipher_ptr[cipher_off];
    const uint64_t cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
    inplace_add_128_128(mult_64_64_128(cipher_val_bx, scalar_val), sum_bx);
    inplace_add_128_128(mult_64_64_128(cipher_val_ax, scalar_val), sum_ax);
  }

  const int64_t out_off = group * cur_limbs * N + limb * N + tid_x;
  res_ptr[out_off] =
      barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
  res_ptr[num_groups * cur_limbs * N + out_off] =
      barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);
}

template <int NUM_GROUPS, int NUM_CIPHER>
__global__ void groupedScalarWeightedAccRegKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ scalar_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN) {
  const int64_t tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (tid_x >= N) {
    return;
  }

  const int64_t limb = blockIdx.y;
  uint128_t sum_bx[NUM_GROUPS];
  uint128_t sum_ax[NUM_GROUPS];
#pragma unroll
  for (int group = 0; group < NUM_GROUPS; ++group) {
    sum_bx[group] = {0, 0};
    sum_ax[group] = {0, 0};
  }

#pragma unroll
  for (int i = 0; i < NUM_CIPHER; ++i) {
    const int64_t cipher_off = i * L_CTN + limb * N + tid_x;
    const uint64_t cipher_val_bx = cipher_ptr[cipher_off];
    const uint64_t cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
#pragma unroll
    for (int group = 0; group < NUM_GROUPS; ++group) {
      const uint64_t scalar_val =
          scalar_ptr[(group * NUM_CIPHER + i) * cur_limbs + limb];
      inplace_add_128_128(
          mult_64_64_128(cipher_val_bx, scalar_val), sum_bx[group]);
      inplace_add_128_128(
          mult_64_64_128(cipher_val_ax, scalar_val), sum_ax[group]);
    }
  }

  const uint64_t mod = mod_ptr[limb];
  const uint64_t barret_ratio = barret_ratio_ptr[limb];
  const uint64_t barret_k = barret_k_ptr[limb];
#pragma unroll
  for (int group = 0; group < NUM_GROUPS; ++group) {
    const int64_t out_off = group * cur_limbs * N + limb * N + tid_x;
    res_ptr[out_off] =
        barret_reduction_128_64(sum_bx[group], mod, barret_ratio, barret_k);
    res_ptr[NUM_GROUPS * cur_limbs * N + out_off] =
        barret_reduction_128_64(sum_ax[group], mod, barret_ratio, barret_k);
  }
}

template <int NUM_GROUPS, int NUM_CIPHER, int X>
__global__ void groupedScalarWeightedAccSharedKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ scalar_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN) {
  static_assert(X <= 128);
  __shared__ uint64_t cipher_bx[NUM_CIPHER][X];
  __shared__ uint64_t cipher_ax[NUM_CIPHER][X];

  const int64_t coeff = blockIdx.x * X + threadIdx.x;
  const int64_t limb = blockIdx.y;
  const int group = threadIdx.y;

  if (threadIdx.y == 0) {
#pragma unroll
    for (int i = 0; i < NUM_CIPHER; ++i) {
      const int64_t cipher_off = i * L_CTN + limb * N + coeff;
      const bool valid = coeff < N;
      cipher_bx[i][threadIdx.x] = valid ? cipher_ptr[cipher_off] : 0;
      cipher_ax[i][threadIdx.x] = valid ? cipher_ptr[cipher_off + BL_CTN] : 0;
    }
  }
  __syncthreads();

  if (coeff >= N) {
    return;
  }

  uint128_t sum_bx = {0, 0};
  uint128_t sum_ax = {0, 0};
#pragma unroll
  for (int i = 0; i < NUM_CIPHER; ++i) {
    const uint64_t scalar_val =
        scalar_ptr[(group * NUM_CIPHER + i) * cur_limbs + limb];
    inplace_add_128_128(
        mult_64_64_128(cipher_bx[i][threadIdx.x], scalar_val), sum_bx);
    inplace_add_128_128(
        mult_64_64_128(cipher_ax[i][threadIdx.x], scalar_val), sum_ax);
  }

  const uint64_t mod = mod_ptr[limb];
  const uint64_t barret_ratio = barret_ratio_ptr[limb];
  const uint64_t barret_k = barret_k_ptr[limb];
  const int64_t out_off = group * cur_limbs * N + limb * N + coeff;
  res_ptr[out_off] =
      barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
  res_ptr[NUM_GROUPS * cur_limbs * N + out_off] =
      barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);
}

__global__ void cpmulBroadcastCipherKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_mu_ptr,
    const int64_t num_ciphers,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t L_PTN) {

  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto bid_y = blockIdx.y;

  auto mod = mod_ptr[bid_y];
  auto barret_mu0 = barret_mu_ptr[bid_y * 2];
  auto barret_mu1 = barret_mu_ptr[bid_y * 2 + 1];
  auto cipher_val_bx = cipher_ptr[bid_y * N + tid_x];
  auto cipher_val_ax = cipher_ptr[bid_y * N + tid_x + L_CTN];


  for (int batch_id = 0; batch_id < num_ciphers; ++batch_id) {
    auto ptx_val = plain_ptr[batch_id * L_PTN + bid_y * N + tid_x];

    auto res_off = batch_id * L_PTN + bid_y * N + tid_x;
    res_ptr[res_off] = mul_mod(cipher_val_bx, ptx_val, mod, barret_mu0, barret_mu1);
    res_ptr[res_off + num_ciphers*cur_limbs*N] =
        mul_mod(cipher_val_ax, ptx_val, mod, barret_mu0, barret_mu1);
  }
}

template <int NB, int NC, int X, int BY>
__global__ void fusedPairwiseMACRegVecKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN,
    const int64_t L_PTN) {
  auto tid_x = blockIdx.x * X + threadIdx.x;
  auto bid_y = blockIdx.y;
  auto lane = threadIdx.y;
  if (tid_x >= N) {
    return;
  }

  uint64_t cbx[NC];
  uint64_t cax[NC];
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    auto cipher_off = i * L_CTN + bid_y * N + tid_x;
    cbx[i] = cipher_ptr[cipher_off];
    cax[i] = cipher_ptr[cipher_off + BL_CTN];
  }

  auto mod = mod_ptr[bid_y];
  auto barret_ratio = barret_ratio_ptr[bid_y];
  auto barret_k = barret_k_ptr[bid_y];
#pragma unroll
  for (int batch_id = lane; batch_id < NB; batch_id += BY) {
    uint128_t sum_bx = {0, 0};
    uint128_t sum_ax = {0, 0};
#pragma unroll
    for (int i = 0; i < NC; ++i) {
      auto plain_val =
          plain_ptr[(batch_id * NC + i) * L_PTN + bid_y * N + tid_x];
      inplace_add_128_128(mult_64_64_128(cbx[i], plain_val), sum_bx);
      inplace_add_128_128(mult_64_64_128(cax[i], plain_val), sum_ax);
    }
    res_ptr[batch_id * cur_limbs * N + bid_y * N + tid_x] =
        barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
    res_ptr
        [NB * cur_limbs * N + batch_id * cur_limbs * N + bid_y * N + tid_x] =
        barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);
  }
}

template <int NB, int NC, int X, int BY, int R>
__global__ void fusedPairwiseMACDirectKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN,
    const int64_t L_PTN) {
  static_assert(BY * R == NB, "direct pairwise MAC expects BY * R == NB");
  auto tid_x = blockIdx.x * X + threadIdx.x;
  auto bid_y = blockIdx.y;
  auto lane = threadIdx.y;
  if (tid_x >= N) {
    return;
  }

  uint128_t sum_bx[R];
  uint128_t sum_ax[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    sum_bx[r] = {0, 0};
    sum_ax[r] = {0, 0};
  }
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    auto cipher_off = i * L_CTN + bid_y * N + tid_x;
    auto cipher_val_bx = cipher_ptr[cipher_off];
    auto cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      auto batch_id = lane * R + r;
      auto plain_val =
          plain_ptr[(batch_id * NC + i) * L_PTN + bid_y * N + tid_x];
      inplace_add_128_128(
          mult_64_64_128(cipher_val_bx, plain_val), sum_bx[r]);
      inplace_add_128_128(
          mult_64_64_128(cipher_val_ax, plain_val), sum_ax[r]);
    }
  }

  auto mod = mod_ptr[bid_y];
  auto barret_ratio = barret_ratio_ptr[bid_y];
  auto barret_k = barret_k_ptr[bid_y];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    auto batch_id = lane * R + r;
    res_ptr[batch_id * cur_limbs * N + bid_y * N + tid_x] =
        barret_reduction_128_64(sum_bx[r], mod, barret_ratio, barret_k);
    res_ptr
        [NB * cur_limbs * N + batch_id * cur_limbs * N + bid_y * N + tid_x] =
        barret_reduction_128_64(sum_ax[r], mod, barret_ratio, barret_k);
  }
}

__global__ void fusedPairwiseMACDirectRuntimeKernel(
    uint64_t* __restrict__ res_ptr,
    const uint64_t* __restrict__ cipher_ptr,
    const uint64_t* __restrict__ plain_ptr,
    const uint64_t* __restrict__ mod_ptr,
    const uint64_t* __restrict__ barret_ratio_ptr,
    const uint64_t* __restrict__ barret_k_ptr,
    const int64_t num_batches,
    const int64_t num_ciphers,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN,
    const int64_t L_PTN) {
  auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  auto bid_y = blockIdx.y;
  auto lane = threadIdx.y;
  if (tid_x >= N) {
    return;
  }

  auto mod = mod_ptr[bid_y];
  auto barret_ratio = barret_ratio_ptr[bid_y];
  auto barret_k = barret_k_ptr[bid_y];
  for (int64_t batch_id = lane; batch_id < num_batches;
       batch_id += blockDim.y) {
    uint128_t sum_bx = {0, 0};
    uint128_t sum_ax = {0, 0};
    for (int64_t i = 0; i < num_ciphers; ++i) {
      auto plain_val =
          plain_ptr[(batch_id * num_ciphers + i) * L_PTN + bid_y * N + tid_x];
      auto cipher_off = i * L_CTN + bid_y * N + tid_x;
      auto cipher_val_bx = cipher_ptr[cipher_off];
      auto cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
      inplace_add_128_128(mult_64_64_128(cipher_val_bx, plain_val), sum_bx);
      inplace_add_128_128(mult_64_64_128(cipher_val_ax, plain_val), sum_ax);
    }
    res_ptr[batch_id * cur_limbs * N + bid_y * N + tid_x] =
        barret_reduction_128_64(sum_bx, mod, barret_ratio, barret_k);
    res_ptr
        [num_batches * cur_limbs * N + batch_id * cur_limbs * N + bid_y * N +
         tid_x] = barret_reduction_128_64(sum_ax, mod, barret_ratio, barret_k);
  }
}

template <int NB, int NC>
void launchPairwiseMACDirectTemplate(
    uint64_t* res_ptr,
    const uint64_t* cipher_ptr,
    const uint64_t* plain_ptr,
    const uint64_t* mod_ptr,
    const uint64_t* barret_ratio_ptr,
    const uint64_t* barret_k_ptr,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN,
    const int64_t L_PTN,
    cudaStream_t stream) {
  constexpr int BY = NB >= 8 ? 8 : NB;
  constexpr int X = BLOCK_SIZE / BY;
  constexpr int R = NB / BY;
  dim3 block(X, BY);
  dim3 grid((N + X - 1) / X, cur_limbs);
  fusedPairwiseMACDirectKernel<NB, NC, X, BY, R><<<grid, block, 0, stream>>>(
      res_ptr,
      cipher_ptr,
      plain_ptr,
      mod_ptr,
      barret_ratio_ptr,
      barret_k_ptr,
      cur_limbs,
      N,
      L_CTN,
      BL_CTN,
      L_PTN);
}

template <int NC>
void launchPairwiseMACReg64Template(
    uint64_t* res_ptr,
    const uint64_t* cipher_ptr,
    const uint64_t* plain_ptr,
    const uint64_t* mod_ptr,
    const uint64_t* barret_ratio_ptr,
    const uint64_t* barret_k_ptr,
    const int64_t cur_limbs,
    const int64_t N,
    const int64_t L_CTN,
    const int64_t BL_CTN,
    const int64_t L_PTN,
    cudaStream_t stream) {
  constexpr int NB = 64;
  constexpr int BY = NC == 9 ? 4 : 8;
  constexpr int X = NC == 9 ? 64 : 32;
  dim3 block(X, BY);
  dim3 grid((N + X - 1) / X, cur_limbs);
  fusedPairwiseMACRegVecKernel<NB, NC, X, BY><<<grid, block, 0, stream>>>(
      res_ptr,
      cipher_ptr,
      plain_ptr,
      mod_ptr,
      barret_ratio_ptr,
      barret_k_ptr,
      cur_limbs,
      N,
      L_CTN,
      BL_CTN,
      L_PTN);
}

} // namespace fhe

namespace at::native {

Tensor batched_pairwise_mac_cuda(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t number_batches,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {
  auto res = at::empty({2, number_batches, cur_limbs, N}, cipher.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, cur_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto L_CTN = cipher.size(2) * N;
  auto BL_CTN = cipher.size(1) * L_CTN;
  auto L_PTN = plaintext.size(2) * N;
  if (number_batches == 1) {
    fhe::fusedPairwiseMACKernel<<<grid, block, 0, stream>>>(
        res.data_ptr<uint64_t>(),
        cipher.data_ptr<uint64_t>(),
        plaintext.data_ptr<uint64_t>(),
        param_primes.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        num_cipher,
        cur_limbs,
        N,
        L_CTN,
        BL_CTN,
        L_PTN);
  } else {
    auto* res_ptr = res.data_ptr<uint64_t>();
    auto* cipher_ptr = cipher.data_ptr<uint64_t>();
    auto* plain_ptr = plaintext.data_ptr<uint64_t>();
    auto* mod_ptr = param_primes.data_ptr<uint64_t>();
    auto* ratio_ptr = barret_ratio.data_ptr<uint64_t>();
    auto* k_ptr = barret_k.data_ptr<uint64_t>();

#define LAUNCH_DIRECT_FOR_NC(NB_VALUE, NC_VALUE)                              \
  fhe::launchPairwiseMACDirectTemplate<NB_VALUE, NC_VALUE>(                   \
      res_ptr,                                                                \
      cipher_ptr,                                                             \
      plain_ptr,                                                              \
      mod_ptr,                                                                \
      ratio_ptr,                                                              \
      k_ptr,                                                                  \
      cur_limbs,                                                              \
      N,                                                                      \
      L_CTN,                                                                  \
      BL_CTN,                                                                 \
      L_PTN,                                                                  \
      stream)

#define DISPATCH_DIRECT_NC(NB_VALUE)                                          \
  switch (num_cipher) {                                                       \
    case 2:                                                                   \
      LAUNCH_DIRECT_FOR_NC(NB_VALUE, 2);                                      \
      break;                                                                  \
    case 4:                                                                   \
      LAUNCH_DIRECT_FOR_NC(NB_VALUE, 4);                                      \
      break;                                                                  \
    case 8:                                                                   \
      LAUNCH_DIRECT_FOR_NC(NB_VALUE, 8);                                      \
      break;                                                                  \
    case 9:                                                                   \
      LAUNCH_DIRECT_FOR_NC(NB_VALUE, 9);                                      \
      break;                                                                  \
    case 16:                                                                  \
      LAUNCH_DIRECT_FOR_NC(NB_VALUE, 16);                                     \
      break;                                                                  \
    case 32:                                                                  \
      LAUNCH_DIRECT_FOR_NC(NB_VALUE, 32);                                     \
      break;                                                                  \
    case 64:                                                                  \
      LAUNCH_DIRECT_FOR_NC(NB_VALUE, 64);                                     \
      break;                                                                  \
    default: {                                                                \
      dim3 runtime_block(32, 8);                                               \
      dim3 runtime_grid((N + 31) / 32, cur_limbs);                            \
      fhe::fusedPairwiseMACDirectRuntimeKernel<<<                             \
          runtime_grid, runtime_block, 0, stream>>>(                          \
          res_ptr,                                                            \
          cipher_ptr,                                                         \
          plain_ptr,                                                          \
          mod_ptr,                                                            \
          ratio_ptr,                                                          \
          k_ptr,                                                              \
          number_batches,                                                     \
          num_cipher,                                                         \
          cur_limbs,                                                          \
          N,                                                                  \
          L_CTN,                                                              \
          BL_CTN,                                                             \
          L_PTN);                                                             \
      break;                                                                  \
    }                                                                         \
  }

#define LAUNCH_REG64_FOR_NC(NC_VALUE)                                         \
  fhe::launchPairwiseMACReg64Template<NC_VALUE>(                              \
      res_ptr,                                                                \
      cipher_ptr,                                                             \
      plain_ptr,                                                              \
      mod_ptr,                                                                \
      ratio_ptr,                                                              \
      k_ptr,                                                                  \
      cur_limbs,                                                              \
      N,                                                                      \
      L_CTN,                                                                  \
      BL_CTN,                                                                 \
      L_PTN,                                                                  \
      stream)

    if (number_batches == 64 && num_cipher <= 64) {
      switch (num_cipher) {
        case 2:
          LAUNCH_REG64_FOR_NC(2);
          break;
        case 4:
          LAUNCH_REG64_FOR_NC(4);
          break;
        case 8:
          LAUNCH_REG64_FOR_NC(8);
          break;
        case 9:
          LAUNCH_REG64_FOR_NC(9);
          break;
        case 16:
          LAUNCH_REG64_FOR_NC(16);
          break;
        case 32:
          LAUNCH_REG64_FOR_NC(32);
          break;
        case 64:
          LAUNCH_REG64_FOR_NC(64);
          break;
        default: {
          dim3 runtime_block(32, 8);
          dim3 runtime_grid((N + 31) / 32, cur_limbs);
          fhe::fusedPairwiseMACDirectRuntimeKernel<<<
              runtime_grid, runtime_block, 0, stream>>>(
              res_ptr,
              cipher_ptr,
              plain_ptr,
              mod_ptr,
              ratio_ptr,
              k_ptr,
              number_batches,
              num_cipher,
              cur_limbs,
              N,
              L_CTN,
              BL_CTN,
              L_PTN);
          break;
        }
      }
    } else {
      switch (number_batches) {
        case 2:
          DISPATCH_DIRECT_NC(2);
          break;
        case 4:
          DISPATCH_DIRECT_NC(4);
          break;
        case 8:
          DISPATCH_DIRECT_NC(8);
          break;
        case 16:
          DISPATCH_DIRECT_NC(16);
          break;
        case 32:
          DISPATCH_DIRECT_NC(32);
          break;
        case 64:
          DISPATCH_DIRECT_NC(64);
          break;
        default: {
          dim3 runtime_block(32, 8);
          dim3 runtime_grid((N + 31) / 32, cur_limbs);
          fhe::fusedPairwiseMACDirectRuntimeKernel<<<
              runtime_grid, runtime_block, 0, stream>>>(
              res_ptr,
              cipher_ptr,
              plain_ptr,
              mod_ptr,
              ratio_ptr,
              k_ptr,
              number_batches,
              num_cipher,
              cur_limbs,
              N,
              L_CTN,
              BL_CTN,
              L_PTN);
          break;
        }
      }
    }

#undef LAUNCH_REG64_FOR_NC
#undef DISPATCH_DIRECT_NC
#undef LAUNCH_DIRECT_FOR_NC
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}


Tensor cpmul_broadcast_pt_cuda(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {

  auto res = at::empty({2, num_cipher, cur_limbs, N}, cipher.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, cur_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto L_CTN = cipher.size(2) * N;
  auto BL_CTN = cipher.size(1) * L_CTN;

  fhe::cpmulBroadcastPTKernel<<<grid, block, 0, stream>>>(
    res.data_ptr<uint64_t>(),
    cipher.data_ptr<uint64_t>(),
    plaintext.data_ptr<uint64_t>(),
    param_primes.data_ptr<uint64_t>(),
    barret_mu.data_ptr<uint64_t>(),
    num_cipher,
    cur_limbs,
    N,
    L_CTN,
    BL_CTN);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;

}

Tensor fused_broadcast_mac_cuda(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t num_plain,
    int64_t cur_limbs,
    int64_t N) {

  auto res = at::empty({2, cur_limbs, N}, cipher.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, cur_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto L_CTN = cipher.size(1) * N;
  auto L_PTN = plaintext.size(2) * N;

  fhe::fusedBroadcastMACKernel<<<grid, block, 0, stream>>>(
      res.data_ptr<uint64_t>(),
      cipher.data_ptr<uint64_t>(),
      plaintext.data_ptr<uint64_t>(),
      param_primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      num_plain,
      cur_limbs,
      N,
      L_CTN,
      L_PTN);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}

Tensor scalar_weighted_acc_cuda(
    const Tensor& cipher,
    const Tensor& scalars,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {

  auto res = at::empty({2, cur_limbs, N}, cipher.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, cur_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto L_CTN = cipher.size(2) * N;
  auto BL_CTN = cipher.size(1) * L_CTN;

  fhe::scalarWeightedAccKernel<<<grid, block, 0, stream>>>(
      res.data_ptr<uint64_t>(),
      cipher.data_ptr<uint64_t>(),
      scalars.data_ptr<uint64_t>(),
      param_primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      num_cipher,
      cur_limbs,
      N,
      L_CTN,
      BL_CTN);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}

Tensor grouped_scalar_weighted_acc_cuda(
    const Tensor& cipher,
    const Tensor& scalars,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t num_groups,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N,
    int64_t strategy) {
  TORCH_CHECK(cipher.is_contiguous(), "cipher must be contiguous");
  TORCH_CHECK(scalars.is_contiguous(), "scalars must be contiguous");
  TORCH_CHECK(num_groups > 0, "num_groups must be positive");
  TORCH_CHECK(num_cipher > 0, "num_cipher must be positive");

  auto res = at::empty({2, num_groups, cur_limbs, N}, cipher.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  auto* res_ptr = res.data_ptr<uint64_t>();
  const auto* cipher_ptr = cipher.data_ptr<uint64_t>();
  const auto* scalar_ptr = scalars.data_ptr<uint64_t>();
  const auto* mod_ptr = param_primes.data_ptr<uint64_t>();
  const auto* ratio_ptr = barret_ratio.data_ptr<uint64_t>();
  const auto* k_ptr = barret_k.data_ptr<uint64_t>();

  const int64_t L_CTN = cipher.size(2) * N;
  const int64_t BL_CTN = cipher.size(1) * L_CTN;

  const int64_t selected_strategy =
      (strategy < 0 && num_cipher == 6 &&
       (num_groups == 6 || num_groups == 7))
      ? 3
      : strategy;

  if (selected_strategy == 1 && num_cipher == 6 && num_groups == 7) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, cur_limbs);
    fhe::groupedScalarWeightedAccRegKernel<7, 6>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else if (selected_strategy == 1 && num_cipher == 6 && num_groups == 6) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, cur_limbs);
    fhe::groupedScalarWeightedAccRegKernel<6, 6>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else if (selected_strategy == 2 && num_cipher == 6 && num_groups == 7) {
    constexpr int X = 64;
    dim3 block(X, 7);
    dim3 grid((N + X - 1) / X, cur_limbs);
    fhe::groupedScalarWeightedAccSharedKernel<7, 6, X>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else if (selected_strategy == 2 && num_cipher == 6 && num_groups == 6) {
    constexpr int X = 64;
    dim3 block(X, 6);
    dim3 grid((N + X - 1) / X, cur_limbs);
    fhe::groupedScalarWeightedAccSharedKernel<6, 6, X>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else if (selected_strategy == 3 && num_cipher == 6 && num_groups == 7) {
    constexpr int X = 32;
    dim3 block(X, 7);
    dim3 grid((N + X - 1) / X, cur_limbs);
    fhe::groupedScalarWeightedAccSharedKernel<7, 6, X>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else if (selected_strategy == 3 && num_cipher == 6 && num_groups == 6) {
    constexpr int X = 32;
    dim3 block(X, 6);
    dim3 grid((N + X - 1) / X, cur_limbs);
    fhe::groupedScalarWeightedAccSharedKernel<6, 6, X>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else if (selected_strategy == 4 && num_cipher == 6 && num_groups == 7) {
    constexpr int X = 128;
    dim3 block(X, 7);
    dim3 grid((N + X - 1) / X, cur_limbs);
    fhe::groupedScalarWeightedAccSharedKernel<7, 6, X>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else if (selected_strategy == 4 && num_cipher == 6 && num_groups == 6) {
    constexpr int X = 128;
    dim3 block(X, 6);
    dim3 grid((N + X - 1) / X, cur_limbs);
    fhe::groupedScalarWeightedAccSharedKernel<6, 6, X>
        <<<grid, block, 0, stream>>>(
            res_ptr,
            cipher_ptr,
            scalar_ptr,
            mod_ptr,
            ratio_ptr,
            k_ptr,
            cur_limbs,
            N,
            L_CTN,
            BL_CTN);
  } else {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, cur_limbs, num_groups);
    fhe::groupedScalarWeightedAccGridKernel<<<grid, block, 0, stream>>>(
        res_ptr,
        cipher_ptr,
        scalar_ptr,
        mod_ptr,
        ratio_ptr,
        k_ptr,
        num_groups,
        num_cipher,
        cur_limbs,
        N,
        L_CTN,
        BL_CTN);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}

Tensor cpmul_broadcast_cipher_cuda(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {

  auto res = at::empty({2, num_cipher, cur_limbs, N}, cipher.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, cur_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto L_CTN = cipher.size(2) * N;
  auto L_PTN = plaintext.size(2) * N;

  fhe::cpmulBroadcastCipherKernel<<<grid, block, 0, stream>>>(
    res.data_ptr<uint64_t>(),
    cipher.data_ptr<uint64_t>(),
    plaintext.data_ptr<uint64_t>(),
    param_primes.data_ptr<uint64_t>(),
    barret_mu.data_ptr<uint64_t>(),
    num_cipher,
    cur_limbs,
    N,
    L_CTN,
    L_PTN);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;

}

} // namespace at::native
