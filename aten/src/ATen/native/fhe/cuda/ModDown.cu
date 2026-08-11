#include <ATen/Dispatch_v2.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/device/Modular.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__global__ void const_sub_mult_batch_kernel(
    uint64_t* to,
    const uint64_t* from,
    const uint64_t* cnst,
    const uint64_t* cnst_psinv,
    const int64_t N,
    const size_t LOG_CV,
    const size_t L_OUTN,
    const size_t BL_OUTN,
    const size_t L_INN,
    const size_t BL_INN,
    const uint64_t* primes) {
  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  from += (cv_id * BL_INN + cipher_id * L_INN);
  to += (cv_id * BL_OUTN + cipher_id * L_OUTN);

  auto prime = primes[blockIdx.y];
  int i = blockIdx.y * N + blockIdx.x * blockDim.x + threadIdx.x;
  auto val = sub_mod(from[i], to[i], prime);
  uint64_t out = mul_and_reduce_shoup(
      val, cnst[blockIdx.y], cnst_psinv[blockIdx.y], prime);
  if (out >= prime)
    out -= prime;
  to[i] = out;
}

__global__ void moddown_kernel(
    uint64_t* to,
    const uint64_t* from,
    size_t N,
    const size_t LOG_CV,
    const size_t L_OUTN,
    const size_t BL_OUTN,
    const size_t L_INN,
    const size_t BL_INN,
    size_t start_length,
    const uint64_t* hat_mod_end,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks) {
  __shared__ uint64_t hat_mod_end_shared[997];
  if (threadIdx.x < start_length) {
    hat_mod_end_shared[threadIdx.x] =
        hat_mod_end[threadIdx.x + blockIdx.y * start_length];
  }
  __syncthreads();

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  from += (cv_id * BL_INN + cipher_id * L_INN);
  to += (cv_id * BL_OUTN + cipher_id * L_OUTN);

  const int degree_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_idx = blockIdx.y;
  uint128_t accum{0};

  for (int i = 0; i < start_length; i++) {
    const uint64_t op1 = from[i * N + degree_idx];
    const uint64_t op2 = hat_mod_end_shared[i]; // out_idx * start_length +
    // const uint64_t op2 = hat_mod_end[out_idx * start_length + i];
    uint128_t out = mult_64_64_128(op1, op2);
    inplace_add_128_128(out, accum);
  }

  const auto prime = primes[out_idx];
  const auto barret_ratio = barret_ratios[out_idx];
  const auto barret_k = barret_ks[out_idx];

  to[out_idx * N + degree_idx] =
      barrett_reduction_128_64(accum, prime, barret_ratio, barret_k);
}

} // namespace fhe

namespace at::native {

static void const_sub_mult_batch(
    uint64_t* out_ptr,
    const uint64_t* op1_ptr,
    const uint64_t* cnst_ptr,
    const uint64_t* cnst_psinv_ptr,
    int64_t batch,
    int64_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* primes_ptr) {
  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256, batch, num_cv * num_cipher);

  auto LOG_CV = (num_cv == 1) ? 0 : 1; // 1 for 2, 0 for 1
  auto L_OUTN = L_OUT * N;
  auto BL_OUTN = L_OUTN * num_cipher;
  auto L_INN = L_IN * N;
  auto BL_INN = L_INN * num_cipher;

  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::const_sub_mult_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out_ptr,
      op1_ptr,
      cnst_ptr,
      cnst_psinv_ptr,
      (int)N,
      LOG_CV,
      L_OUTN,
      BL_OUTN,
      L_INN,
      BL_INN,
      primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void moddown_impl(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    size_t sizeP,
    size_t end_length,
    const Tensor& primes,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  auto LOG_CV = (num_cv == 1) ? 0 : 1; // 1 for 2, 0 for 1
  auto L_OUTN = L_OUT * N;
  auto BL_OUTN = L_OUTN * num_cipher;
  auto L_INN = L_IN * N;
  auto BL_INN = L_INN * num_cipher;

  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown;
  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256, end_length, num_cv * num_cipher);
  auto ptr = from_ptr + N * end_length;
  auto primes_ptr = primes.data_ptr<uint64_t>();
  auto param_barret_ratio_ptr = barret_ratio.data_ptr<uint64_t>();
  auto param_barret_k_ptr = barret_k.data_ptr<uint64_t>();
  auto prod_q_i_mod_q_j_ptr = prod_q_i_mod_q_j.data_ptr<uint64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::moddown_kernel<<<grid_dim, block_dim, 0, stream>>>(
      to_ptr,
      ptr,
      N,
      LOG_CV,
      L_OUTN,
      BL_OUTN,
      L_INN,
      BL_INN,
      sizeP,
      prod_q_i_mod_q_j_ptr,
      primes_ptr,
      param_barret_ratio_ptr,
      param_barret_k_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void moddown_cuda_template(
    Tensor& res,
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
    const Tensor& inverse_scaled_power_of_roots_div_two,
    Tensor& workspace) {
  const int start_length = sizeP;
  const int end_length = curr_limbs;

  auto num_cv = 1;
  auto batch = from.sizes()[0];

  auto L_IN = from.sizes()[1];
  auto L_OUT = res.sizes()[1];

  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  iNTT_impl(
      workspace_ptr + curr_limbs * N,
      from_ptr + curr_limbs * N,
      start_length,
      N,
      L_IN,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>() + L,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + L * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() + L * N);

  const_mult_batch(
      workspace_ptr + curr_limbs * N,
      workspace_ptr + curr_limbs * N,
      hat_inverse_vec_moddown.data_ptr<uint64_t>(),
      hat_inverse_vec_shoup_moddown.data_ptr<uint64_t>(),
      sizeP,
      N,
      L_IN,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>() + L);

  moddown_impl(
      to_ptr,
      workspace_ptr,
      N,
      L_OUT,
      L_IN,
      num_cv,
      batch,
      sizeP,
      end_length,
      primes,
      prod_q_i_mod_q_j_moddown,
      barret_ratio,
      barret_k);

  NTT_impl(
      to_ptr,
      end_length,
      N,
      L_OUT,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  const_sub_mult_batch(
      to_ptr,
      from_ptr,
      prod_inv_moddown.data_ptr<uint64_t>(),
      prod_inv_shoup_moddown.data_ptr<uint64_t>(),
      end_length,
      N,
      L_OUT,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>());

    // vsub_mod(
    //     num_cv,
    //     batch,
    //     L_OUT,
    //     L_IN,
    //     L_OUT,
    //     N,
    //     end_length,
    //     to_ptr,
    //     from_ptr,
    //     to_ptr,
    //     primes.data_ptr<uint64_t>());

    // const_mult_batch(
    //     to_ptr,
    //     to_ptr,
    //     prod_inv_moddown.data_ptr<uint64_t>(),
    //     prod_inv_shoup_moddown.data_ptr<uint64_t>(),
    //     end_length,
    //     N,
    //     L_OUT,
    //     L_OUT,
    //     num_cv,
    //     batch,
    //     primes.data_ptr<uint64_t>());
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
  TORCH_INTERNAL_ASSERT(in.dim() == 3);
  TORCH_INTERNAL_ASSERT(in.sizes()[1] == (curr_limbs + sizeP));
  auto batch = in.sizes()[0];

  auto out = at::empty({batch, curr_limbs, N}, in.options());
  auto workspace =
      at::empty({batch, curr_limbs + sizeP, N}, in.options());

  moddown_cuda_template(
      out,
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
      inverse_scaled_power_of_roots_div_two,
      workspace);
  return out;
}

Tensor moddown_write_cuda(
    const Tensor& out,
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
  TORCH_INTERNAL_ASSERT(in.dim() == 3);
  TORCH_INTERNAL_ASSERT(out.dim() == 3);
  TORCH_INTERNAL_ASSERT(in.sizes()[1] == (curr_limbs + sizeP));
  TORCH_INTERNAL_ASSERT(out.sizes()[0] == in.sizes()[0]);
  TORCH_INTERNAL_ASSERT(out.sizes()[1] == curr_limbs);
  TORCH_INTERNAL_ASSERT(out.sizes()[2] == N);

  auto mutable_out = out;
  auto workspace =
      at::empty({in.sizes()[0], curr_limbs + sizeP, N}, in.options());

  moddown_cuda_template(
      mutable_out,
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
      inverse_scaled_power_of_roots_div_two,
      workspace);
  return out;
}

} // namespace at::native
