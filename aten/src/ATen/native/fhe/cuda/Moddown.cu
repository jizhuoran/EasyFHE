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

#include <ATen/native/fhe/cuda/arithmetic.h>
#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__global__ void moddown_kernel(
    uint64_t* to,
    const uint64_t* ptr,
    const int64_t N,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks,
    const uint64_t* hat_mod_end,
    const uint64_t start_length) { // it should be the size of the Auxiliary CRT
  // basis {P} = {p_1,...,p_k}

  const int degree_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_idx = blockIdx.y;
  uint128_t accum{0};

  for (int i = 0; i < start_length; i++) {
    const uint64_t op1 = ptr[i * N + degree_idx];
    const uint64_t op2 = hat_mod_end[out_idx * start_length + i];
    uint128_t out = mult_64_64_128(op1, op2);
    inplace_add_128_128(out, accum);
  }

  const auto prime = primes[out_idx];
  const auto barret_ratio = barret_ratios[out_idx];
  const auto barret_k = barret_ks[out_idx];

  to[out_idx * N + degree_idx] =
      barret_reduction_128_64(accum, prime, barret_ratio, barret_k);
}

} // namespace fhe

namespace at::native {

static void moddown_impl(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    const int64_t N,
    const int64_t sizeP,
    const int64_t start_length,
    const int64_t end_length,
    const Tensor& primes,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown[0];

  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256, end_length);
  auto ptr = from_ptr + N * end_length;
  auto primes_ptr = primes.data_ptr<uint64_t>();
  auto param_barret_ratio_ptr = barret_ratio.data_ptr<uint64_t>();
  auto param_barret_k_ptr = barret_k.data_ptr<uint64_t>();
  auto prod_q_i_mod_q_j_ptr = prod_q_i_mod_q_j.data_ptr<uint64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::moddown_kernel<<<
      grid_dim,
      block_dim,
      0,
      stream>>>(
      to_ptr,
      ptr,
      N,
      primes_ptr,
      param_barret_ratio_ptr,
      param_barret_k_ptr,
      prod_q_i_mod_q_j_ptr,
      sizeP);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void moddown_cuda_template(
    Tensor& res,
    Tensor& workspace,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    int64_t log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  const int start_length = sizeP;
  const int end_length = curr_limbs;

  auto hat_inverse_vec = hat_inverse_vec_moddown[0];
  auto hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown[0];

  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  iNTT_impl(
      from_ptr,
      workspace_ptr,
      end_length,
      start_length,
      curr_limbs,
      L,
      N,
      inverse_power_of_roots_div_two,
      primes,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch(
      workspace_ptr + curr_limbs * N,
      workspace_ptr + curr_limbs * N,
      hat_inverse_vec.data_ptr<uint64_t>(),
      hat_inverse_vec_psinv.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>() + L,
      sizeP,
      N);

  moddown_impl(
      to_ptr,
      workspace_ptr,
      N,
      sizeP,
      start_length,
      end_length,
      primes,
      prod_q_i_mod_q_j_moddown,
      barret_ratio,
      barret_k);

  NTT_impl(
      to_ptr,
      to_ptr,
      end_length,
      N,
      power_of_roots_shoup.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  const auto& prod_inv = prod_inv_moddown[0];
  const auto& prod_inv_psinv = prod_inv_shoup_moddown[0];

  vsub_mod(
      N, end_length, to_ptr, to_ptr, from_ptr, primes.data_ptr<uint64_t>());
  vneg_mod(N, end_length, to_ptr, to_ptr, nullptr, primes.data_ptr<uint64_t>());

  const_mult_batch(
      to_ptr,
      to_ptr,
      prod_inv.data_ptr<uint64_t>(),
      prod_inv_psinv.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      end_length,
      N);
}

Tensor moddown_cuda(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    int64_t log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  auto out = at::empty(curr_limbs * N, in.options());
  auto workspace = at::empty((curr_limbs + sizeP) * N, in.options());
  moddown_cuda_template(
      out,
      workspace,
      in,
      curr_limbs,
      L,
      sizeP,
      N,
      log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two);
  return out;
}

} // namespace at::native