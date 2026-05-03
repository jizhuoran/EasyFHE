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
#include <ATen/TensorIndexing.h>
#include <ATen/ops/cat.h>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__global__ void modup_step_two_kernel(
    uint64_t* to,
    const uint64_t* from,
    const int64_t begin_idx,
    const int64_t N,
    const int64_t alpha,
    const int64_t curr_limbs,
    const int64_t L,
    const uint64_t group_size,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
    const uint64_t* hat_mod_end) {
  const int degree_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   const int hat_mod_end_idx = blockIdx.y;


    __shared__ uint64_t hat_mod_end_shared[997];
    if(threadIdx.x < group_size) {
      hat_mod_end_shared[threadIdx.x] = hat_mod_end[threadIdx.x + blockIdx.y * alpha];
    }
    __syncthreads();


  uint128_t accum{0};
  for (int i = 0; i < group_size; i++) {
    const uint64_t op1 = from[i * N + degree_idx];
    const uint64_t op2 = hat_mod_end_shared[i]; //blockIdx.y * alpha + 
    // const uint64_t op2 = hat_mod_end[blockIdx.y * alpha + i];
    uint128_t out = mult_64_64_128(op1, op2);
    inplace_add_128_128(out, accum);
  }

  auto gap = L - curr_limbs;
  const int out_idx =
      blockIdx.y + ((blockIdx.y >= begin_idx) ? group_size : 0);
  auto prime_idx = out_idx +
      (((out_idx >= 0 && out_idx < begin_idx) ||
        (out_idx >= (begin_idx + group_size) && out_idx < curr_limbs))
           ? 0
           : gap);  // add a gap to index special modulus, since we save Q_L*P in primes
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
  const auto begin_idx = beta_idx * alpha;
  auto group_size =
      ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;
  const auto end_length = curr_limbs + sizeP - group_size;

  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256 / 1, end_length);

//   std::cout << "modup_matmul: "
//             << "alpha: " << alpha
//             << ", beta_idx: " << beta_idx
//             << ", begin_idx: " << begin_idx
//             << ", group_size: " << group_size
//             << ", end_length: " << end_length
//             << ", N: " << N
//             << ", curr_limbs: " << curr_limbs
//             << ", L: " << L
//             << ", sizeQP: " << sizeQP
//             << ", sizeP: " << sizeP
//             << ", L_OUT: " << L_OUT
//             << ", L_IN: " << L_IN
//             << std::endl;

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
      group_size,
      primes_ptr,
      barret_ratio_ptr,
      barret_k_ptr,
      prod_q_i_mod_q_j_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void modup_cuda_template(
    Tensor& to,
    const Tensor& from,
    int64_t cur_limbs,
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
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup) {
  auto num_cipher = from.sizes()[1];
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  int64_t num_moduli_after_modup = cur_limbs + sizeP;

  auto L_OUT = to.sizes()[2];
  auto L_IN = from.sizes()[2];

  uint64_t* to_ptr__ = reinterpret_cast<uint64_t*>(to.data_ptr<uint64_t>());
  uint64_t* from_ptr__ = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());

  for (size_t group_idx = 0; group_idx < beta; ++group_idx) {
    auto to_ptr_ = to_ptr__ + (num_moduli_after_modup * N * group_idx);
    auto from_ptr_ = from_ptr__ + (alpha * N * group_idx);

    size_t begin_idx = group_idx * alpha;
    size_t in_C_L_len =
        ((begin_idx + alpha) > cur_limbs) ? (cur_limbs - begin_idx) : alpha;

    auto hat_inverse_vec =
        hat_inverse_vecs[group_idx * alpha + (in_C_L_len - 1)];
    auto hat_inverse_vec_psinv =
        hat_inverse_vec_shoups[group_idx * alpha + (in_C_L_len - 1)];

    auto stream = at::cuda::getCurrentCUDAStream();

    iNTT_impl(
        to_ptr_ + begin_idx * N,
        from_ptr_,
        in_C_L_len,
        N,
        L_OUT,
        L_IN,
        1, // num_cv
        num_cipher,
        primes.data_ptr<uint64_t>() + begin_idx,
        inverse_power_of_roots_div_two.data_ptr<uint64_t>() + begin_idx * N,
        inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() +
            begin_idx * N);

    const_mult_batch(
        to_ptr_ + begin_idx * N,
        to_ptr_ + begin_idx * N,
        hat_inverse_vec.data_ptr<uint64_t>(),
        hat_inverse_vec_psinv.data_ptr<uint64_t>(),
        in_C_L_len,
        N,
        L_OUT,
        L_OUT,
        1, // num_cv
        num_cipher,
        primes.data_ptr<uint64_t>() + begin_idx);

    for (int cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      auto to_ptr = to_ptr_ + (L_OUT * N * cipher_id);
      auto from_ptr = from_ptr_ + (L_IN * N * cipher_id);

      modup_matmul(
          to_ptr,
          to_ptr + N * begin_idx,
          group_idx,
          alpha,
          N,
          cur_limbs,
          L,
          prod_q_i_mod_q_js,
          primes,
          barret_ratio,
          barret_k);
    }

    if (begin_idx > 0) {
      NTT_impl(
          to_ptr_,
          begin_idx,
          N,
          L_OUT,
          1, // num_cv
          num_cipher,
          primes.data_ptr<uint64_t>(),
          power_of_roots_shoup.data_ptr<uint64_t>(),
          power_of_roots.data_ptr<uint64_t>());
    }
    if (cur_limbs - begin_idx - in_C_L_len > 0) {
      NTT_impl(
          to_ptr_ + (begin_idx + in_C_L_len) * N,
          cur_limbs - begin_idx - in_C_L_len,
          N,
          L_OUT,
          1, // num_cv
          num_cipher,

          primes.data_ptr<uint64_t>() + begin_idx + in_C_L_len,
          power_of_roots_shoup.data_ptr<uint64_t>() +
              (begin_idx + in_C_L_len) * N,
          power_of_roots.data_ptr<uint64_t>() + (begin_idx + in_C_L_len) * N);
    }
    if (sizeP > 0) {
      NTT_impl(
          to_ptr_ + cur_limbs * N,
          sizeP,
          N,
          L_OUT,
          1, // num_cv
          num_cipher,

          primes.data_ptr<uint64_t>() + L,
          power_of_roots_shoup.data_ptr<uint64_t>() + L * N,
          power_of_roots.data_ptr<uint64_t>() + L * N);
    }

    for (int cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      cudaMemcpyAsync(
          to_ptr_ + (L_OUT * N * cipher_id) + N * begin_idx,
          from_ptr_ + (L_IN * N * cipher_id),
          8 * in_C_L_len * N,
          cudaMemcpyDeviceToDevice,
          stream);
    }
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
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  auto num_cv = in.sizes()[0]; // should be 1
  TORCH_INTERNAL_ASSERT(num_cv == 1);
  auto batch = in.sizes()[1];

  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out =
      at::empty({num_cv, batch, beta * (curr_limbs + sizeP), N}, in.options());

  modup_cuda_template(
      out, // out_ptr + beta * (curr_limbs + sizeP) * N * batch_id,
      in, // in_ptr + in.sizes()[2] * N * batch_id,
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
      power_of_roots,
      power_of_roots_shoup);

  return out;
}

} // namespace at::native
