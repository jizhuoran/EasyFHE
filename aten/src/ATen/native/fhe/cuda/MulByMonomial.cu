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

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {
__global__ void mulByMonomialKernel_step1(
    const uint64_t* in,
    const uint64_t* qVec,
    uint64_t* out,
    const int N) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  out[blockIdx.y * N + tid_x] = qVec[blockIdx.y] - in[blockIdx.y * N + tid_x];
}

__global__ void mulByMonomialKernel_step2(
    uint64_t* out,
    const uint64_t* qVec,
    const uint64_t* in,
    const int N,
    const int shift) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (tid_x < shift) {
    out[blockIdx.y * N + tid_x] =
        qVec[blockIdx.y] - in[blockIdx.y * N + tid_x + (N - shift)];
  } else {
    out[blockIdx.y * N + tid_x] = in[blockIdx.y * N + tid_x - shift];
  }
}

__global__ void mulByMonomialKernel_step1_step2(
    uint64_t* out,
    const uint64_t* qVec,
    const uint64_t* in,
    const int N,
    const int shift) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (tid_x < shift) {
    out[blockIdx.y * N + tid_x] = in[blockIdx.y * N + tid_x + (N - shift)];
  } else {
    out[blockIdx.y * N + tid_x] =
        (qVec[blockIdx.y] - in[blockIdx.y * N + tid_x - shift]);
  }
}

} // namespace fhe

namespace at::native {

static void mul_by_monomial_impl(
    uint64_t* out_ptr,
    const uint64_t* primes_ptr,
    const uint64_t* in_ptr,
    const int64_t l,
    const int64_t N,
    const int64_t M,
    const int64_t monomialDeg) {

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, l);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto shift = monomialDeg % M;
  if (shift < N) {
    fhe::mulByMonomialKernel_step2<<<grid, block, 0, stream>>>(
      out_ptr, primes_ptr, in_ptr, N, shift);
  } else {
    shift = shift % N;
    fhe::mulByMonomialKernel_step1_step2<<<grid, block, 0, stream>>>(
        out_ptr, primes_ptr, in_ptr, N, shift);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void mul_by_monomial_inplace_template(
    Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {

  auto res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      res_ptr,
      res_ptr,
      0,
      l,
      l,
      level,
      N,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  Tensor temp = at::empty_like(res);
  auto temp_ptr = reinterpret_cast<uint64_t*>(temp.data_ptr<uint64_t>());
  auto param_primes_ptr = reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
  mul_by_monomial_impl(temp_ptr, param_primes_ptr, res_ptr, l, N, M, monomialDeg);
  cudaMemcpy(res_ptr, temp_ptr, l * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

  NTT_impl(
      res_ptr,
      res_ptr,
      l,
      N,
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_primes.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());
}

Tensor mul_by_monomial_cuda(
    const Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  
  TORCH_INTERNAL_ASSERT(false, "mul_by_monomial_cuda only supports inplace operation");
  return res;
}

Tensor& mul_by_monomial_cuda_(
    Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  mul_by_monomial_inplace_template(
      res,
      param_primes,
      l,
      N,
      M,
      monomialDeg,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return res;
}

Tensor& mul_by_monomial_cuda_out(
    const Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    Tensor& out) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented");
  return out;
}

} // namespace at::native