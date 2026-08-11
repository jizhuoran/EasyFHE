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

#include "ATen/native/fhe/cuda/device/Launch.cuh"
#include "ATen/native/fhe/cuda/device/Modular.cuh"
#include "ATen/native/fhe/cuda/CommonOperation.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {
__global__ void const_mult_batch_kernel(
    uint64_t* __restrict__ to,
    const uint64_t* __restrict__ from,
    const uint64_t* __restrict__ cnst,
    const uint64_t* __restrict__ cnst_psinv,
    const int64_t N,
    const size_t num_cipher,
    const size_t L_OUTN,
    const size_t BL_OUTN,
    const size_t L_INN,
    const size_t BL_INN,
    const uint64_t* __restrict__ primes) {
  const int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (tid >= N) {
    return;
  }

  const size_t cv_cipher = blockIdx.z;
  const size_t cv_id = cv_cipher / num_cipher;
  const size_t cipher_id = cv_cipher - cv_id * num_cipher;
  const int64_t limb = blockIdx.y;
  const int64_t index = limb * N + tid;

  from += cv_id * BL_INN + cipher_id * L_INN;
  to += cv_id * BL_OUTN + cipher_id * L_OUTN;

  const uint64_t prime = primes[limb];
  uint64_t out =
      mul_and_reduce_shoup(from[index], cnst[limb], cnst_psinv[limb], prime);
  if (out >= prime)
    out -= prime;
  to[index] = out;
}



// note: SwitchModulus in mubintvecnat.cpp (align with update in openFHE commit:
// 64fd8426, 07/14/23)
// DANGER change last modulus operation to a minues
__global__ void switch_modulus_kernel(
    uint64_t* __restrict__ to,
    const uint64_t* __restrict__ from,
    const int64_t N,
    const uint64_t old_modulus_by_two,
    const size_t num_cipher,
    const size_t L_OUTN,
    const size_t BL_OUTN,
    const size_t L_INN,
    const size_t BL_INN,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ diffs) {
  const int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (tid >= N) {
    return;
  }

  const size_t cv_cipher = blockIdx.z;
  const size_t cv_id = cv_cipher / num_cipher;
  const size_t cipher_id = cv_cipher - cv_id * num_cipher;
  const int64_t limb = blockIdx.y;

  from += cv_id * BL_INN + cipher_id * L_INN;
  to += cv_id * BL_OUTN + cipher_id * L_OUTN;

  const uint64_t in_val = from[tid];
  uint64_t res = in_val + (in_val > old_modulus_by_two ? diffs[limb] : 0);
  const uint64_t new_modulus = primes[limb];
  if (res >= new_modulus) {
    res -= new_modulus;
  }
  to[limb * N + tid] = res;
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
  TORCH_INTERNAL_ASSERT(num_cv > 0, "const_mult_batch expects num_cv > 0");
  TORCH_INTERNAL_ASSERT(num_cipher > 0, "const_mult_batch expects num_cipher > 0");
  auto block_dim = dim3(BLOCK_SIZE);
  auto grid_dim = dim3(num_blocks(N), batch, num_cv * num_cipher);

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
      N,
      num_cipher,
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
  TORCH_INTERNAL_ASSERT(num_cv > 0, "switch_modulus expects num_cv > 0");
  TORCH_INTERNAL_ASSERT(num_cipher > 0, "switch_modulus expects num_cipher > 0");
  auto primes_ptr =
      reinterpret_cast<const uint64_t*>(primes.data_ptr<uint64_t>());
  auto switch_modulus_map_ptr =
      reinterpret_cast<const uint64_t*>(switch_modulus_map.data_ptr<uint64_t>());

  auto L_OUTN = L_OUT * N;
  auto BL_OUTN = L_OUTN * num_cipher;
  auto L_INN = L_IN * N;
  auto BL_INN = L_INN * num_cipher;

  auto block_dim = dim3(BLOCK_SIZE);
  auto grid_dim = dim3(num_blocks(N), batch, num_cv * num_cipher);

  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::switch_modulus_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out_ptr,
      in_ptr,
      N,
      old_modulus_by_two,
      num_cipher,
      L_OUTN,
      BL_OUTN,
      L_INN,
      BL_INN,
      primes_ptr,
      switch_modulus_map_ptr + old_prime_index * primes.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

}

} // namespace at::native
