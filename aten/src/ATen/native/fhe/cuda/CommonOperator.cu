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

#include "ATen/native/fhe/cuda/Utils.cuh"
#include "ATen/native/fhe/cuda/CommonOperation.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {
__global__ void const_mult_batch_kernel(
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
  uint64_t out = mul_and_reduce_shoup(from[i], cnst[blockIdx.y], cnst_psinv[blockIdx.y], prime);
  if (out >= prime)
    out -= prime;
  to[i] = out;
}



// note: SwitchModulus in mubintvecnat.cpp (align with update in openFHE commit:
// 64fd8426, 07/14/23)
// DANGER change last modulus operation to a minues
__global__ void switch_modulus_kernel(
    uint64_t* to,
    const uint64_t* from,
    const int64_t N,
    const uint64_t old_modulus_by_two,
    const size_t LOG_CV,
    const size_t L_OUTN,
    const size_t BL_OUTN,
    const size_t L_INN,
    const size_t BL_INN,
    const uint64_t* primes,
    const uint64_t* diffs) {
  

      auto cipher_id = blockIdx.z >> LOG_CV;
      auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);

      from += (cv_id * BL_INN + cipher_id * L_INN);
      to += (cv_id * BL_OUTN + cipher_id * L_OUTN);

  int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto in_val = from[input_idx];

  auto res = in_val + (in_val > old_modulus_by_two? diffs[blockIdx.y] : 0);
  auto new_modulus = primes[blockIdx.y];
  if (res >= new_modulus) {
    res -= new_modulus;
  }
  to[blockIdx.y * N + input_idx] = res;

}


} // namespace fhe

namespace at::native {

void const_mult_batch(
    uint64_t* out_ptr,
    const uint64_t* op1_ptr,
    const uint64_t* op2_ptr,
    const uint64_t* op2_psinv_ptr,
    size_t batch,
    size_t N,
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
  fhe::const_mult_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out_ptr,
      op1_ptr,
      op2_ptr,
      op2_psinv_ptr,
      (int)N,
      LOG_CV,
      L_OUTN,
      BL_OUTN,
      L_INN,
      BL_INN,
      primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



void switch_modulus(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    int64_t old_prime_index,
    int64_t batch,
    int64_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    uint64_t old_modulus_by_two,
    const Tensor& primes,
    const Tensor& switch_modulus_map) {
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto switch_modulus_map_ptr = reinterpret_cast<uint64_t*>(switch_modulus_map.data_ptr<uint64_t>());

  auto LOG_CV = (num_cv == 1) ? 0 : 1; // 1 for 2, 0 for 1
  auto L_OUTN = L_OUT * N;
  auto BL_OUTN = L_OUTN * num_cipher;
  auto L_INN = L_IN * N;
  auto BL_INN = L_INN * num_cipher;


  auto block_dim = dim3(256);
  auto grid_dim = dim3(N / 256, batch, num_cv*num_cipher);

  // N * batch / block_dim;
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::switch_modulus_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out_ptr,
      in_ptr,
      N,
      old_modulus_by_two,
      LOG_CV,
      L_OUTN,
      BL_OUTN,
      L_INN,
      BL_INN,
      primes_ptr,
      switch_modulus_map_ptr + old_prime_index * primes.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

}

} // namespace at::native
