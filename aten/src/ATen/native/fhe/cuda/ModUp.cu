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
#include "ATen/native/fhe/cuda/device/Launch.cuh"
#include "ATen/native/fhe/cuda/device/Modular.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__global__ void modup_step_two_kernel(
    uint64_t* __restrict__ to,
    const uint64_t* __restrict__ from,
    const int64_t begin_idx,
    const int64_t N,
    const int64_t alpha,
    const int64_t curr_limbs,
    const int64_t L,
    const int64_t group_size,
    const int64_t L_OUTN,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barrett_ratios,
    const uint64_t* __restrict__ barrett_ks,
    const uint64_t* __restrict__ hat_mod_end) {
  const int degree_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t out_iter = blockIdx.y;
  const int64_t cipher_id = blockIdx.z;
  extern __shared__ uint64_t hat_mod_end_shared[];
  __shared__ int64_t out_idx_shared;
  __shared__ uint64_t prime_shared;
  __shared__ uint64_t barret_ratio_shared;
  __shared__ uint64_t barret_k_shared;

  if (threadIdx.x == 0) {
    const int64_t gap = L - curr_limbs;
    const int64_t out_idx =
        out_iter + ((out_iter >= begin_idx) ? group_size : 0);
    // out_idx has already skipped [begin_idx, begin_idx + group_size), so Q
    // limbs map to the same prime index and P limbs jump over the dropped Q
    // tail.
    const int64_t prime_idx = out_idx + ((out_idx < curr_limbs) ? 0 : gap);
    out_idx_shared = out_idx;
    prime_shared = primes[prime_idx];
    barret_ratio_shared = barrett_ratios[prime_idx];
    barret_k_shared = barrett_ks[prime_idx];
  }
  if (threadIdx.x < group_size) {
    hat_mod_end_shared[threadIdx.x] =
        hat_mod_end[threadIdx.x + out_iter * alpha];
  }
  __syncthreads();

  if (degree_idx >= N) {
    return;
  }

  to += cipher_id * L_OUTN;
  from += cipher_id * L_OUTN;

  uint128_t accum{0};
  for (int i = 0; i < group_size; i++) {
    const uint64_t op1 = from[i * N + degree_idx];
    const uint64_t op2 = hat_mod_end_shared[i];
    uint128_t out = mult_64_64_128(op1, op2);
    inplace_add_128_128(out, accum);
  }

  to[out_idx_shared * N + degree_idx] = barrett_reduction_128_64(
      accum, prime_shared, barret_ratio_shared, barret_k_shared);
}

__global__ void modup_const_mult_all_kernel(
    uint64_t* __restrict__ data,
    const int64_t N,
    const int64_t alpha,
    const int64_t curr_limbs,
    const int64_t LN,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ hat_inverse_vecs,
    const uint64_t* __restrict__ hat_inverse_vec_shoups,
    const int64_t hat_stride) {
  const int64_t degree_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  const int64_t cipher_id = blockIdx.z;
  if (degree_idx >= N || limb >= curr_limbs) {
    return;
  }

  const int64_t begin_idx = (limb / alpha) * alpha;
  const int64_t group_size =
      min(alpha, curr_limbs - begin_idx);
  const int64_t local_idx = limb - begin_idx;
  const int64_t hat_row = begin_idx + group_size - 1;
  const uint64_t prime = primes[limb];
  const uint64_t scalar = hat_inverse_vecs[hat_row * hat_stride + local_idx];
  const uint64_t scalar_shoup =
      hat_inverse_vec_shoups[hat_row * hat_stride + local_idx];

  uint64_t* out = data + cipher_id * LN + limb * N + degree_idx;
  uint64_t value = mul_and_reduce_shoup(*out, scalar, scalar_shoup, prime);
  if (value >= prime) {
    value -= prime;
  }
  *out = value;
}

__global__ void modup_copy_original_all_kernel(
    uint64_t* __restrict__ to,
    const uint64_t* __restrict__ from,
    const int64_t N,
    const int64_t alpha,
    const int64_t curr_limbs,
    const int64_t num_moduli_after_modup,
    const int64_t L_OUTN,
    const int64_t L_INN) {
  const int64_t degree_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  const int64_t cipher_id = blockIdx.z;
  if (degree_idx >= N || limb >= curr_limbs) {
    return;
  }

  const int64_t beta_idx = limb / alpha;
  const int64_t physical_limb = beta_idx * num_moduli_after_modup + limb;
  to[cipher_id * L_OUTN + physical_limb * N + degree_idx] =
      from[cipher_id * L_INN + limb * N + degree_idx];
}

} // namespace fhe

namespace at::native {
static void modup_matmul(
    uint64_t* to_ptr,
    const uint64_t* from_ptr,
    int64_t beta_idx,
    const int64_t alpha,
    const int64_t N,
    int64_t curr_limbs,
    int64_t L,
    int64_t L_OUT,
    int64_t num_cipher,
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

  auto block_dim = dim3(BLOCK_SIZE);
  auto grid_dim = dim3(num_blocks(N), end_length, num_cipher);
  auto L_OUTN = L_OUT * N;

  auto primes_ptr =
      reinterpret_cast<const uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<const uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<const uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto prod_q_i_mod_q_j_ptr =
      reinterpret_cast<const uint64_t*>(
          prod_q_i_mod_q_js.data_ptr<uint64_t>()) +
      beta_idx * prod_q_i_mod_q_js.stride(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::modup_step_two_kernel<<<
      grid_dim,
      block_dim,
      group_size * sizeof(uint64_t),
      stream>>>(
      to_ptr,
      from_ptr,
      begin_idx,
      N,
      alpha,
      curr_limbs,
      L,
      group_size,
      L_OUTN,
      primes_ptr,
      barret_ratio_ptr,
      barret_k_ptr,
      prod_q_i_mod_q_j_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void modup_cuda_template(
    Tensor& to,
    const Tensor& from,
    const Tensor* temp_workspace,
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
    const Tensor& power_of_roots_shoup,
    bool copy_original_limbs) {
  auto num_cipher = from.sizes()[0];
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  int64_t num_moduli_after_modup = cur_limbs + sizeP;

  auto L_OUT = to.sizes()[1];
  auto L_IN = from.sizes()[1];

  uint64_t* to_ptr__ = reinterpret_cast<uint64_t*>(to.data_ptr<uint64_t>());
  uint64_t* from_ptr__ = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (beta > 1) {
    Tensor temp_storage;
    const Tensor* temp_tensor = temp_workspace;
    if (temp_tensor == nullptr) {
      temp_storage = at::empty({num_cipher, cur_limbs, N}, from.options());
      temp_tensor = &temp_storage;
    } else {
      TORCH_INTERNAL_ASSERT(temp_tensor->dim() == 3);
      TORCH_INTERNAL_ASSERT(temp_tensor->sizes()[0] == num_cipher);
      TORCH_INTERNAL_ASSERT(temp_tensor->sizes()[1] == cur_limbs);
      TORCH_INTERNAL_ASSERT(temp_tensor->sizes()[2] == N);
      TORCH_CHECK(
          temp_tensor->is_contiguous(),
          "modup temp workspace must be contiguous");
    }
    auto temp = *temp_tensor;
    auto* temp_ptr = reinterpret_cast<uint64_t*>(temp.data_ptr<uint64_t>());

    iNTT_modup_scaled_impl(
        temp_ptr,
        from_ptr__,
        cur_limbs,
        N,
        cur_limbs,
        L_IN,
        1, // num_cv
        num_cipher,
        alpha,
        primes.data_ptr<uint64_t>(),
        inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
        inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>(),
        hat_inverse_vecs.data_ptr<uint64_t>(),
        hat_inverse_vec_shoups.data_ptr<uint64_t>(),
        hat_inverse_vecs.stride(0));

    modup_step_two_ntt_all_impl(
        to_ptr__,
        temp_ptr,
        beta,
        cur_limbs,
        N,
        L,
        alpha,
        num_moduli_after_modup,
        L_OUT,
        cur_limbs,
        1, // num_cv
        num_cipher,
        primes.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        prod_q_i_mod_q_js.data_ptr<uint64_t>(),
        prod_q_i_mod_q_js.stride(0),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>());

    if (copy_original_limbs) {
      fhe::modup_copy_original_all_kernel<<<
          dim3(num_blocks(N), cur_limbs, num_cipher),
          BLOCK_SIZE,
          0,
          stream>>>(
          to_ptr__,
          from_ptr__,
          N,
          alpha,
          cur_limbs,
          num_moduli_after_modup,
          L_OUT * N,
          L_IN * N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return;
  }

  for (int64_t group_idx = 0; group_idx < beta; ++group_idx) {
    auto to_ptr_ = to_ptr__ + (num_moduli_after_modup * N * group_idx);
    auto from_ptr_ = from_ptr__ + (alpha * N * group_idx);

    int64_t begin_idx = group_idx * alpha;
    int64_t in_C_L_len =
        ((begin_idx + alpha) > cur_limbs) ? (cur_limbs - begin_idx) : alpha;

    const int64_t hat_row = group_idx * alpha + (in_C_L_len - 1);
    const auto* hat_inverse_vec_ptr =
        hat_inverse_vecs.data_ptr<uint64_t>() +
        hat_row * hat_inverse_vecs.stride(0);
    const auto* hat_inverse_vec_psinv_ptr =
        hat_inverse_vec_shoups.data_ptr<uint64_t>() +
        hat_row * hat_inverse_vec_shoups.stride(0);

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
        hat_inverse_vec_ptr,
        hat_inverse_vec_psinv_ptr,
        in_C_L_len,
        N,
        L_OUT,
        L_OUT,
        1, // num_cv
        num_cipher,
        primes.data_ptr<uint64_t>() + begin_idx);

    modup_matmul(
        to_ptr_,
        to_ptr_ + N * begin_idx,
        group_idx,
        alpha,
        N,
        cur_limbs,
        L,
        L_OUT,
        num_cipher,
        prod_q_i_mod_q_js,
        primes,
        barret_ratio,
        barret_k);

    NTT_modup_masked_impl(
        to_ptr_,
        num_moduli_after_modup,
        cur_limbs,
        N,
        L,
        begin_idx,
        in_C_L_len,
        L_OUT,
        1, // num_cv
        num_cipher,
        primes.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>());

    if (copy_original_limbs) {
      C10_CUDA_CHECK(cudaMemcpy2DAsync(
          to_ptr_ + N * begin_idx,
          L_OUT * N * sizeof(uint64_t),
          from_ptr_,
          L_IN * N * sizeof(uint64_t),
          in_C_L_len * N * sizeof(uint64_t),
          num_cipher,
          cudaMemcpyDeviceToDevice,
          stream));
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
  TORCH_INTERNAL_ASSERT(in.dim() == 3);
  auto batch = in.sizes()[0];

  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out =
      at::empty({batch, beta * (curr_limbs + sizeP), N}, in.options());

  modup_cuda_template(
      out, // out_ptr + beta * (curr_limbs + sizeP) * N * batch_id,
      in, // in_ptr + in.sizes()[2] * N * batch_id,
      nullptr,
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
      power_of_roots_shoup,
      true);

  return out;
}

void modup_without_copy_cuda_out(
    Tensor& out,
    const Tensor& temp_workspace,
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
  TORCH_INTERNAL_ASSERT(in.dim() == 3);
  auto batch = in.sizes()[0];

  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  TORCH_INTERNAL_ASSERT(out.dim() == 3);
  TORCH_INTERNAL_ASSERT(out.sizes()[0] == batch);
  TORCH_INTERNAL_ASSERT(out.sizes()[1] == beta * (curr_limbs + sizeP));
  TORCH_INTERNAL_ASSERT(out.sizes()[2] == N);
  TORCH_CHECK(out.is_contiguous(), "modup output workspace must be contiguous");

  modup_cuda_template(
      out,
      in,
      &temp_workspace,
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
      power_of_roots_shoup,
      false);
}

Tensor modup_without_copy_cuda(
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
  TORCH_INTERNAL_ASSERT(in.dim() == 3);
  auto batch = in.sizes()[0];

  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out =
      at::empty({batch, beta * (curr_limbs + sizeP), N}, in.options());

  modup_cuda_template(
      out,
      in,
      nullptr,
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
      power_of_roots_shoup,
      false);

  return out;
}

} // namespace at::native
