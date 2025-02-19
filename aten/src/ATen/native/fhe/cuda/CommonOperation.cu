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
    uint64_t* op1,
    const uint64_t* op2,
    const uint64_t* op2_psinv,
    const uint64_t* primes,
    const size_t N,
    const int start_prime_idx,
    const int batch,
    const int start_op1_idx,
    const int start_op2_idx) {
  STRIDED_LOOP_START(N * batch, i);
  const int op2_idx = start_op2_idx + i / N;
  const int prime_idx = i / N + start_prime_idx;
  const auto prime = primes[prime_idx];

  uint64_t out = mul_and_reduce_shoup(
      op1[start_op1_idx * N + i], op2[op2_idx], op2_psinv[op2_idx], prime);

  if (out >= prime)
    out -= prime;
  to[start_op1_idx * N + i] = out;
  STRIDED_LOOP_END;
}
__global__ void vec_add_mod_batch_kernel(
    uint64_t* to,
    const uint64_t* op1,
    const uint64_t* op2,
    const int N,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks,
    const uint64_t batch) {
  STRIDED_LOOP_START((N * batch), i);
  const int out_prime_idx = i / N;
  const auto prime = primes[out_prime_idx];
  const auto barret_ratio = barret_ratios[out_prime_idx];
  const auto barret_k = barret_ks[out_prime_idx];
  barret_reduction_64_64(op1[i] + op2[i], to[i], prime, barret_ratio, barret_k);

  STRIDED_LOOP_END;
}

__global__ void vec_mod_batch_kernel(
    uint64_t* to,
    const uint64_t* op1,
    const int N,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks,
    const uint64_t batch) {
  STRIDED_LOOP_START((N * batch), i);
  const int out_prime_idx = i / N;
  const int op1_idx = i % N;
  const auto prime = primes[out_prime_idx];
  const auto barret_ratio = barret_ratios[out_prime_idx];
  const auto barret_k = barret_ks[out_prime_idx];
  barret_reduction_64_64(op1[op1_idx], to[i], prime, barret_ratio, barret_k);

  STRIDED_LOOP_END;
}

// note: SwitchModulus in mubintvecnat.cpp (align with update in openFHE commit:
// 64fd8426, 07/14/23)
__global__ void switch_modulus_kernel(
    uint64_t* to,
    const uint64_t* ptr,
    const size_t N,
    const size_t batch,
    const size_t old_prime_idx,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks) {
  STRIDED_LOOP_START(batch * N, i)
  auto old_modulus_by_two = primes[old_prime_idx] >> 1;
  auto old_modulus = primes[old_prime_idx];
  auto new_modulus_idx = i / N;
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
  int input_idx = i % N;
  auto tmp = (ptr[input_idx] > old_modulus_by_two) ? diff : 0;

  if (new_modulus >= old_modulus) {
    to[i] = tmp + ptr[input_idx];
  } else { // old_modulus > new_modulus
    to[i] = tmp + ptr[input_idx];
    if (to[i] >= new_modulus)
      barret_reduction_64_64(to[i], to[i], new_modulus, barret_ratio, barret_k);
  }
  STRIDED_LOOP_END;
}

} // namespace fhe

namespace at::native {
void const_mult_batch(
    uint64_t* out_ptr,
    uint64_t* op1_ptr,
    const Tensor& op2,
    const Tensor& op2_psinv,
    const Tensor& primes,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t start_op1_idx,
    int64_t start_op2_idx,
    int64_t N) {
  AT_DISPATCH_V2(
      op2.scalar_type(),
      "const_mult_batch",
      AT_WRAP([&]() {
        auto op2_ptr = reinterpret_cast<uint64_t*>(op2.data_ptr<uint64_t>());
        auto op2_psinv_ptr =
            reinterpret_cast<uint64_t*>(op2_psinv.data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = N * batch / block_dim;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::const_mult_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
            out_ptr,
            op1_ptr,
            op2_ptr,
            op2_psinv_ptr,
            primes_ptr,
            (int)N,
            (int)start_prime_idx,
            (int)batch,
            (int)start_op1_idx,
            (int)start_op2_idx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

void vec_add_mod_batch(
    uint64_t* out_ptr,
    uint64_t* in1_ptr,
    uint64_t* in2_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t batch,
    int64_t N) {
  AT_DISPATCH_V2(
      primes.scalar_type(),
      "vec_add_mod_batch",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = N * batch / block_dim;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::vec_add_mod_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
            out_ptr,
            in1_ptr,
            in2_ptr,
            (int)N,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            (int)batch);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

void vec_mod_batch(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t batch,
    int64_t N) {
  AT_DISPATCH_V2(
      primes.scalar_type(),
      "vec_mod_batch",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = N * batch / block_dim;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::vec_mod_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
            out_ptr,
            in_ptr,
            (int)N,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            (int)batch);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

void switch_modulus(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t old_prime_index,
    int64_t batch,
    int64_t N) {
  AT_DISPATCH_V2(
      primes.scalar_type(),
      "switch_modulus",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = N * batch / block_dim;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::switch_modulus_kernel<<<grid_dim, block_dim, 0, stream>>>(
            out_ptr,
            in_ptr,
            (int)N,
            batch,
            old_prime_index,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}
} // namespace at::native