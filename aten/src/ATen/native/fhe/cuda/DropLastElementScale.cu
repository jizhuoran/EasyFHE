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

__global__ void const_mult_add_batch_kernel(
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
  uint64_t out = mul_and_reduce_shoup(
      from[i], cnst[blockIdx.y], cnst_psinv[blockIdx.y], prime);
  if (out >= prime)
    out -= prime;
  to[i] = add_mod(to[i], out, prime);
}

} // namespace fhe

namespace at::native {

static void const_mult_add_batch(
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
  fhe::const_mult_add_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
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

static void rescale_one_level_template(
    Tensor& res,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    uint64_t old_primes,
    const Tensor& param_primes,
    const Tensor& switch_modulus_map,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    Tensor& workspace) {
  const int end_length = curr_limbs - 1;

  const int num_cv = 1;
  const int num_cipher = from.sizes()[0];
  const int L_IN = from.sizes()[1];
  const int L_OUT = res.sizes()[1];

  auto from_ptr_ = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto workspace_ptr_ =
      reinterpret_cast<uint64_t*>(workspace.data_ptr<uint64_t>());
  auto to_ptr_ = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  iNTT_impl(
      workspace_ptr_ + N * end_length,
      from_ptr_ + N * end_length,
      1,
      N,
      L_IN,
      L_IN,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>() + end_length,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + end_length * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() +
          end_length * N);

  switch_modulus(
      to_ptr_,
      workspace_ptr_ + N * end_length,
      curr_limbs - 1,
      curr_limbs - 1,
      N,
      L_OUT,
      L_IN,
      num_cv,
      num_cipher,
      old_primes >> 1,
      param_primes,
      switch_modulus_map);

  int start_op2_idx = (L - curr_limbs) * (L - 1);
  const_mult_batch(
      to_ptr_,
      to_ptr_,
      qlql_inv_mod_ql_div_ql_mod_q.data_ptr<uint64_t>() + start_op2_idx,
      qlql_inv_mod_ql_div_ql_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
      curr_limbs - 1,
      N,
      L_OUT,
      L_OUT,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>());

  NTT_impl(
      to_ptr_,
      end_length,
      N,
      L_OUT,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());
  start_op2_idx = (curr_limbs - 1) * (L);

  const_mult_add_batch(
      to_ptr_,
      from_ptr_,
      q_inv_mod_q.data_ptr<uint64_t>() + start_op2_idx,
      q_inv_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
      end_length,
      N,
      L_OUT,
      L_IN,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>());
}

Tensor rescale_one_level_cuda(
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    int64_t old_primes,
    const Tensor& param_primes,
    const Tensor& switch_modulus_map,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  TORCH_INTERNAL_ASSERT(from.dim() == 3);
  auto batch = from.sizes()[0];

  auto res = at::empty({batch, (curr_limbs - 1), N}, from.options());
  auto workspace = at::empty({batch, curr_limbs, N}, from.options());

  rescale_one_level_template(
      res,
      from,
      curr_limbs,
      l,
      L,
      N,
      old_primes,
      param_primes,
      switch_modulus_map,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      workspace);
  return res;
}

} // namespace at::native
