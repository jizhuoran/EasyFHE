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
#include "ATen/native/fhe/cuda/NttImpl.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {
__global__ void const_mult_batch_kernel(
    uint64_t* to,
    const uint64_t* op1,
    const uint64_t* op2,
    const uint64_t* op2_psinv,
    const int64_t N,
    const int64_t batch,
    const uint64_t* primes) {
  const int op2_idx = blockIdx.y;
  const int prime_idx = blockIdx.y;
  const auto prime = primes[prime_idx];

  int i = blockIdx.y * N + blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t out =
      mul_and_reduce_shoup(op1[i], op2[op2_idx], op2_psinv[op2_idx], prime);

  if (out >= prime)
    out -= prime;
  to[i] = out;
}

// note: SwitchModulus in mubintvecnat.cpp (align with update in openFHE commit:
// 64fd8426, 07/14/23)
__global__ void switch_modulus_kernel(
    uint64_t* to,
    const uint64_t* ptr,
    const int64_t old_prime_idx,
    const int64_t batch,
    const int64_t N,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks) {
  auto old_modulus_by_two = primes[old_prime_idx] >> 1;
  auto old_modulus = primes[old_prime_idx];
  auto new_modulus_idx = blockIdx.y;
  auto new_modulus = primes[new_modulus_idx];
  auto barret_ratio = barret_ratios[new_modulus_idx];
  auto barret_k = barret_ks[new_modulus_idx];
  uint64_t diff;
  if (old_modulus > new_modulus) {
    uint64_t temp_out;
    barret_reduction_64_64(
        old_modulus, temp_out, new_modulus, barret_ratio, barret_k);
    diff = new_modulus - temp_out;
  } else {
    diff = new_modulus - old_modulus;
  }
  int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto tmp = (ptr[input_idx] > old_modulus_by_two) ? diff : 0;

  int i = new_modulus_idx * N + input_idx;
  if (new_modulus >= old_modulus) {
    to[i] = tmp + ptr[input_idx];
  } else { // old_modulus > new_modulus
    to[i] = tmp + ptr[input_idx];
    if (to[i] >= new_modulus)
      barret_reduction_64_64(to[i], to[i], new_modulus, barret_ratio, barret_k);
  }
}

} // namespace fhe

namespace at::native {

void const_mult_batch(
    uint64_t* out_ptr,
    const uint64_t* op1_ptr,
    const uint64_t* op2_ptr,
    const uint64_t* op2_psinv_ptr,
    int64_t batch,
    int64_t N,
    const uint64_t* primes_ptr) {
  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256, batch);
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::const_mult_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out_ptr, op1_ptr, op2_ptr, op2_psinv_ptr, (int)N, (int)batch, primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void switch_modulus(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    int64_t old_prime_index,
    int64_t batch,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256, batch);
  // N * batch / block_dim;
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::switch_modulus_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out_ptr,
      in_ptr,
      old_prime_index,
      batch,
      N,
      primes_ptr,
      barret_ratio_ptr,
      barret_k_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace at::native