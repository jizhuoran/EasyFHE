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
#include <ATen/native/fhe/cuda/modupdown.h>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {
// __device__ uint128_t4 accumulate_in_modup(
//     const uint64_t* ptr,
//     const int N,
//     const uint64_t* hat_mod_end,
//     const int start_length,
//     const int degree_idx,
//     const int hat_mod_end_idx) {
//   uint128_t4 accum{0};
//   for (int i = 0; i < start_length; i++) {
//     const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
//     uint128_t4 out;
//   ulonglong4 op1;
//   op1 = *reinterpret_cast<const ulonglong4*>(ptr + i * N + degree_idx);

//     out.x = mult_64_64_128(op1.x, op2);
//     inplace_add_128_128(out.x, accum.x);
//     out.y = mult_64_64_128(op1.y, op2);
//     inplace_add_128_128(out.y, accum.y);
//     out.z = mult_64_64_128(op1.z, op2);
//     inplace_add_128_128(out.z, accum.z);
//     out.w = mult_64_64_128(op1.w, op2);
//     inplace_add_128_128(out.w, accum.w);
//   }
//   return accum;
// }

__global__ void modup_step_two_kernel(
    uint64_t* to,
    const uint64_t* ptr,
    const int begin_idx,
    const int N,
    const int alpha,
    const int curr_limbs,
    const int L,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
    const uint64_t* hat_mod_end,
    const int hat_mod_end_size,
    const uint64_t start_length,
    const uint64_t end_length) {
  constexpr const int unroll_number = 4;
  extern __shared__ uint64_t s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  const int degree_idx = unroll_number * (blockIdx.x * blockDim.x + threadIdx.x);
  const int hat_mod_end_idx = blockIdx.y;

  const int out_idx =
      hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);
  uint128_t4 accum = accumulate_in_modupdown(
      ptr, N, s_hat_mod_end, alpha, degree_idx, hat_mod_end_idx);
  int gap = L - curr_limbs;
  int prime_idx = out_idx +
      (((out_idx >= 0 && out_idx < begin_idx) ||
        (out_idx >= (begin_idx + start_length) && out_idx < curr_limbs))
           ? 0
           : gap);
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barrett_ratios[prime_idx];
  const auto barret_k = barrett_ks[prime_idx];

  ulonglong4 out;
    out.x = barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    out.y = barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    out.z = barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    out.w = barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    *reinterpret_cast<ulonglong4*>(&to[out_idx * N + degree_idx]) = out;
}

} // namespace fhe

namespace at::native {
static void modup_matmul(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    int64_t beta_idx,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const int64_t alpha,
    const int64_t N,
    const Tensor& prod_q_i_mod_q_js,
    int64_t curr_limbs,
    int64_t L) {
  const int unroll_factor = 4;
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int begin_idx = (int)beta_idx * (int)alpha;
  int start_length =
      ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;
  const int end_length = curr_limbs + sizeP - start_length;
//   int grid_dim{(int)N * end_length / 256 / unroll_factor};
//   int block_dim{256};

  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256 / unroll_factor, end_length);

  const auto& prod_q_i_mod_q_j = prod_q_i_mod_q_js[beta_idx];


        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        auto prod_q_i_mod_q_j_ptr =
            reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::modup_step_two_kernel<<<
            grid_dim,
            block_dim,
            prod_q_i_mod_q_j.size(-1) * sizeof(uint64_t),
            stream>>>(
            to_ptr,
            from_ptr,
            begin_idx,
            N,
            alpha,
            curr_limbs,
            L,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            prod_q_i_mod_q_j_ptr,
            prod_q_i_mod_q_j.size(-1),
            start_length,
            end_length);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void modup_impl(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    int idx,
    int curr_limbs,
    int L,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const int64_t N,
    const int64_t alpha,
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
      inverse_power_of_roots_div_two,
      primes,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch(
      to_ptr,
      to_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      primes,
      begin_idx,
      in_C_L_len,
      begin_idx,
      0,
      N);

  modup_matmul(
      to_ptr,
      to_ptr + N * begin_idx,
      idx,
      primes,
      barret_ratio,
      barret_k,
      alpha,
      N,
      prod_q_i_mod_q_js,
      curr_limbs,
      L);

  NTT_except_some_range_impl(
      to_ptr,
      0,
      num_moduli_after_modup,
      N,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      L,
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
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t beta,
    int64_t N,
    int64_t alpha,
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
        hat_inverse_vecs,
        hat_inverse_vec_shoups,
        N,
        alpha,
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
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t beta,
    int64_t N,
    int64_t alpha,
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
      hat_inverse_vecs,
      hat_inverse_vec_shoups,
      prod_q_i_mod_q_js,
      primes,
      barret_ratio,
      barret_k,
      beta,
      N,
      alpha,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      power_of_roots_shoup,
      power_of_roots);
  return out;
}

} // namespace at::native