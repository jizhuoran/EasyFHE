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
    fhe::fusedPairwiseMACKernel_batch<<<
        grid,
        block,
        num_cipher * BLOCK_SIZE * sizeof(uint64_t) * 2,
        stream>>>(
        res.data_ptr<uint64_t>(),
        cipher.data_ptr<uint64_t>(),
        plaintext.data_ptr<uint64_t>(),
        param_primes.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        number_batches,
        num_cipher,
        cur_limbs,
        N,
        L_CTN,
        BL_CTN,
        L_PTN);
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
