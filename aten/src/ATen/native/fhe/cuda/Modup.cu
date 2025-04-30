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

__global__ void modup_step_two_kernel(
    uint64_t* to,
    const uint64_t* ptr,
    const int begin_idx,
    const int N,
    const int alpha,
    const int curr_limbs,
    const int L,
    const uint64_t start_length,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
    const uint64_t* hat_mod_end) {
  const int degree_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int hat_mod_end_idx = blockIdx.y;

  const int out_idx =
      hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);

  uint128_t accum{0};
  for (int i = 0; i < start_length; i++) {
    const uint64_t op1 = ptr[i * N + degree_idx];
    const uint64_t op2 = hat_mod_end[hat_mod_end_idx * alpha + i];
    uint128_t out = mult_64_64_128(op1, op2);
    inplace_add_128_128(out, accum);
  }

  int gap = L - curr_limbs;
  int prime_idx = out_idx +
      (((out_idx >= 0 && out_idx < begin_idx) ||
        (out_idx >= (begin_idx + start_length) && out_idx < curr_limbs))
           ? 0
           : gap);
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barrett_ratios[prime_idx];
  const auto barret_k = barrett_ks[prime_idx];

  to[out_idx * N + degree_idx] =
      barret_reduction_128_64(accum, prime, barret_ratio, barret_k);
}

} // namespace fhe

namespace at::native {
static void modup_matmul(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    int64_t beta_idx,
    const int64_t alpha,
    const int64_t N,
    int64_t curr_limbs,
    int64_t L,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int begin_idx = (int)beta_idx * (int)alpha;
  int start_length =
      ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;
  const int end_length = curr_limbs + sizeP - start_length;

  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256 / 1, end_length);

  const auto& prod_q_i_mod_q_j = prod_q_i_mod_q_js[beta_idx];

  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto prod_q_i_mod_q_j_ptr =
      reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::modup_step_two_kernel<<<grid_dim, block_dim, 0, stream>>>(
      to_ptr,
      from_ptr,
      begin_idx,
      N,
      alpha,
      curr_limbs,
      L,
      start_length,
      primes_ptr,
      barret_ratio_ptr,
      barret_k_ptr,
      prod_q_i_mod_q_j_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void modup_impl(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    int idx,
    int curr_limbs,
    int L,
    const int64_t N,
    const int64_t alpha,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  int num_moduli_after_modup = curr_limbs + sizeP;
  size_t begin_idx = idx * alpha;
  size_t in_C_L_len =
      ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;

  auto hat_inverse_vec = hat_inverse_vecs[idx * alpha + (in_C_L_len - 1)];
  auto hat_inverse_vec_psinv =
      hat_inverse_vec_shoups[idx * alpha + (in_C_L_len - 1)];

  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(
      to_ptr + (N * begin_idx),
      from_ptr,
      8 * in_C_L_len * N,
      cudaMemcpyDeviceToDevice,
      stream);

  iNTT_impl(
      to_ptr,
      to_ptr,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      L,
      N,
      primes,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch(
      to_ptr + begin_idx * N,
      to_ptr + begin_idx * N,
      hat_inverse_vec.data_ptr<uint64_t>(),
      hat_inverse_vec_psinv.data_ptr<uint64_t>(),
      in_C_L_len,
      N,
      primes.data_ptr<uint64_t>() + begin_idx);

  modup_matmul(
      to_ptr,
      to_ptr + N * begin_idx,
      idx,
      alpha,
      N,
      curr_limbs,
      L,
      prod_q_i_mod_q_js,
      primes,
      barret_ratio,
      barret_k);

  NTT_except_some_range_impl(
      to_ptr,
      num_moduli_after_modup,
      N,
      curr_limbs,
      L,
      0,
      begin_idx,
      in_C_L_len,
      power_of_roots_shoup,
      primes,
      power_of_roots);

  cudaMemcpyAsync(
      to_ptr + N * begin_idx,
      from_ptr,
      8 * in_C_L_len * N,
      cudaMemcpyDeviceToDevice,
      stream);
}

static void modup_cuda_template(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    int64_t curr_limbs,
    int64_t L,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  int num_moduli_after_modup = curr_limbs + sizeP;
  for (int i = 0; i < beta; ++i) {
    modup_impl(
        out_ptr + (num_moduli_after_modup * N) * i,
        in_ptr + (alpha * N * i),
        i,
        curr_limbs,
        L,
        N,
        alpha,
        hat_inverse_vecs,
        hat_inverse_vec_shoups,
        primes,
        barret_ratio,
        barret_k,
        prod_q_i_mod_q_js,
        inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two,
        power_of_roots_shoup,
        power_of_roots);
  }
}

Tensor modup_cuda(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out = at::empty(beta * (curr_limbs + sizeP) * N, in.options());
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  modup_cuda_template(
      out_ptr,
      in_ptr,
      curr_limbs,
      L,
      beta,
      N,
      alpha,
      hat_inverse_vecs,
      hat_inverse_vec_shoups,
      prod_q_i_mod_q_js,
      primes,
      barret_ratio,
      barret_k,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      power_of_roots_shoup,
      power_of_roots);
  return out;
}

} // namespace at::native