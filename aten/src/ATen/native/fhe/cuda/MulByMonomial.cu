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

__global__ void mul_by_monomial_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ in,
    const uint64_t* __restrict__ qVec,
    const int64_t N,
    const int64_t L,
    const int64_t active_limbs,
    const int64_t num_cipher,
    const int64_t shift,
    const bool negated_wrap) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (tid_x >= N) {
    return;
  }

  const int64_t limb = blockIdx.y;
  const int64_t cv_cipher = blockIdx.z;
  const int64_t cv_id = cv_cipher / num_cipher;
  const int64_t cipher_id = cv_cipher - cv_id * num_cipher;
  const int64_t in_base = (cv_id * num_cipher + cipher_id) * L * N + limb * N;
  const int64_t out_base =
      (cv_id * num_cipher + cipher_id) * active_limbs * N + limb * N;

  const bool wrapped = tid_x < shift;
  const int64_t src = wrapped ? tid_x + (N - shift) : tid_x - shift;
  const uint64_t in_val = in[in_base + src];
  if (wrapped == negated_wrap) {
    out[out_base + tid_x] = qVec[limb] - in_val;
  } else {
    out[out_base + tid_x] = in_val;
  }
}


} // namespace fhe

namespace at::native {

static void mul_by_monomial_impl(
    uint64_t* out_ptr,
    const uint64_t* in_ptr,
    const uint64_t* primes_ptr,
    const int64_t num_cv,
    const int64_t num_cipher,
    const int64_t l,
    const int64_t L,
    const int64_t N,
    const int64_t M,
    const int64_t monomialDeg,
    cudaStream_t stream) {

  dim3 block(BLOCK_SIZE);
  dim3 grid(num_blocks(N), l, num_cv * num_cipher);
  auto shift = monomialDeg % M;
  if (shift < 0) {
    shift += M;
  }
  const bool negated_wrap = shift < N;
  shift %= N;
  fhe::mul_by_monomial_kernel<<<grid, block, 0, stream>>>(
      out_ptr,
      in_ptr,
      primes_ptr,
      N,
      L,
      l,
      num_cipher,
      shift,
      negated_wrap);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void mul_by_monomial_inplace_template(
    Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  auto num_cv = res.sizes()[0];
  auto num_cipher = res.sizes()[1];
  auto L = res.sizes()[2];
  auto LN = res.sizes()[2] * N;
  auto BLN = LN * num_cipher;
  auto active_LN = l * N;
  auto active_BLN = active_LN * num_cipher;

  auto res_ptr_ = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  auto stream = at::cuda::getCurrentCUDAStream();

  iNTT_impl(
      res_ptr_,
      res_ptr_,
      l,
      N,
      L,
      L,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>(),
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());

  Tensor temp = at::empty({num_cv, num_cipher, l, N}, res.options());
  auto temp_ptr = reinterpret_cast<uint64_t*>(temp.data_ptr<uint64_t>());
  auto param_primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
  mul_by_monomial_impl(
      temp_ptr,
      res_ptr_,
      param_primes_ptr,
      num_cv,
      num_cipher,
      l,
      L,
      N,
      M,
      monomialDeg,
      stream);

  for (int64_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (int64_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      cudaMemcpyAsync(
          res_ptr_ + cv_id * BLN + cipher_id * LN,
          temp_ptr + cv_id * active_BLN + cipher_id * active_LN,
          active_LN * sizeof(uint64_t),
          cudaMemcpyDeviceToDevice,
          stream);
    }
  }

  NTT_impl(
      res_ptr_,
      l,
      N,
      L,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());
}

Tensor mul_by_monomial_cuda(
    const Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  
  TORCH_INTERNAL_ASSERT(false, "mul_by_monomial_cuda only supports inplace operation");
  return res;
}

Tensor& mul_by_monomial_cuda_(
    Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  TORCH_INTERNAL_ASSERT(res.dim() == 4);
  TORCH_INTERNAL_ASSERT(res.sizes()[0] > 0);


  mul_by_monomial_inplace_template(
      res,
      l,
      N,
      M,
      monomialDeg,
      level,
      param_primes,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);

  return res;
}

Tensor& mul_by_monomial_cuda_out(
    const Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    Tensor& out) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented");
  return out;
}

} // namespace at::native
