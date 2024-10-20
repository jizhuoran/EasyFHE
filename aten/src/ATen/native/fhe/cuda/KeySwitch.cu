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

#include "ATen/native/fhe/cuda/NttImpl.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__global__ void modup_step_two_simple(
    const uint64_t* ptr_after_intt,
    const uint64_t* ptr_before_intt,
    const int in_prime_idx,
    const int degree,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_Ks,
    const uint64_t end_length,
    uint64_t* to) {
  STRIDED_LOOP_START(degree * end_length, i);
  const int out_prime_idx = i / degree;
  const int degree_idx = i % degree;
  const auto barret_ratio = barrett_ratios[out_prime_idx];
  const auto barret_k = barrett_Ks[out_prime_idx];
  if (out_prime_idx != in_prime_idx) {
    const auto in = ptr_after_intt[degree_idx];
    if (primes[in_prime_idx] > primes[out_prime_idx]) {
      barret_reduction_64_64(
          in, to[i], primes[out_prime_idx], barret_ratio, barret_k);
    } else {
      to[i] = in;
    }
  } else {
    to[i] = ptr_before_intt[degree_idx];
  }
  STRIDED_LOOP_END;
}

__global__ void const_mult_batch(
    size_t degree,
    const uint64_t* primes,
    const uint64_t* op1,
    const uint64_t* op2,
    const uint64_t* op2_psinv,
    const int start_prime_idx,
    const int batch,
    const int start_op1_idx,
    uint64_t* to) {
  STRIDED_LOOP_START(degree * batch, i);
  const int op2_idx = i / degree;
  const int prime_idx = op2_idx + start_prime_idx;
  const auto prime = primes[prime_idx];
  uint64_t out = mul_and_reduce_shoup(
      op1[start_op1_idx * degree + i], op2[op2_idx], op2_psinv[op2_idx], prime);
  if (out >= prime)
    out -= prime;
  to[start_op1_idx * degree + i] = out;
  STRIDED_LOOP_END;
}

__device__ uint128_t4 accumulate_in_modup(
    const uint64_t* ptr,
    const int degree,
    const uint64_t* hat_mod_end,
    const int start_length,
    const int degree_idx,
    const int hat_mod_end_idx) {
  uint128_t4 accum{0};
  for (int i = 0; i < start_length; i++) {
    const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
    uint128_t4 out;
    uint64_t op1_x, op1_y, op1_z, op1_w;
    asm("{\n\t"
        "ld.global.v2.u64 {%0, %1}, [%2];\n\t"
        "}"
        : "=l"(op1_x), "=l"(op1_y)
        : "l"(ptr + i * degree + degree_idx));

    out.x = mult_64_64_128(op1_x, op2);
    inplace_add_128_128(out.x, accum.x);
    out.y = mult_64_64_128(op1_y, op2);
    inplace_add_128_128(out.y, accum.y);
    asm("{\n\t"
        "ld.global.v2.u64 {%0, %1}, [%2];\n\t"
        "}"
        : "=l"(op1_z), "=l"(op1_w)
        : "l"(ptr + i * degree + degree_idx + 2));
    out.z = mult_64_64_128(op1_z, op2);
    inplace_add_128_128(out.z, accum.z);
    out.w = mult_64_64_128(op1_w, op2);
    inplace_add_128_128(out.w, accum.w);
  }
  return accum;
}

__global__ void modup_step_two_kernel(
    const uint64_t* ptr,
    const int begin_idx,
    const int degree,
    const int alpha,
    const int curr_limbs,
    const int level,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_Ks,
    const uint64_t* hat_mod_end,
    const int hat_mod_end_size,
    const uint64_t start_length,
    const uint64_t end_length,
    uint64_t* to) {
  constexpr const int unroll_number = 4;
  extern __shared__ uint64_t s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  STRIDED_LOOP_START(
      (degree * end_length + unroll_number - 1) / unroll_number, i);
  const int degree_idx = unroll_number * (i / end_length);
  const int hat_mod_end_idx = i % end_length;
  const int out_idx =
      hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);
  uint128_t4 accum = accumulate_in_modup(
      ptr, degree, s_hat_mod_end, alpha, degree_idx, hat_mod_end_idx);
  int gap = level - end_length;
  int prime_idx = out_idx +
      (((out_idx >= 0 && out_idx < begin_idx) ||
        (out_idx >= (begin_idx + start_length) && out_idx < curr_limbs))
           ? 0
           : gap);
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barrett_ratios[prime_idx];
  const auto barret_k = barrett_Ks[prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_idx * degree + degree_idx),
        "l"(out),
        "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_idx * degree + degree_idx + 2),
        "l"(out),
        "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void moddown_kernel(
    int degree_,
    uint64_t* d_primes,
    uint64_t* d_barret_ratio,
    uint64_t* d_barret_k,
    int log_degree_,
    const uint64_t* ptr,
    const uint64_t* hat_mod_end,
    const int hat_mod_end_size,
    const uint64_t start_length,
    const uint64_t end_length,
    uint64_t* to) {
  constexpr const int unroll_number = 4;
  extern __shared__ uint64_t s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  STRIDED_LOOP_START(
      (degree_ * end_length + unroll_number - 1) / unroll_number, i);
  const int degree_idx = unroll_number * (i / end_length);
  const int out_prime_idx = i % end_length;
  uint128_t4 accum = accumulate_in_modup(
      ptr, degree_, s_hat_mod_end, start_length, degree_idx, out_prime_idx);
  const auto prime = d_primes[out_prime_idx];
  const auto barret_ratio = d_barret_ratio[out_prime_idx];
  const auto barret_k = d_barret_k[out_prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * degree_ + degree_idx),
        "l"(out),
        "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * degree_ + degree_idx + 2),
        "l"(out),
        "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void negateInplace_(
    size_t degree,
    size_t log_degree,
    size_t batch,
    const uint64_t* primes,
    uint64_t* op) {
  STRIDED_LOOP_START(batch * degree, i);
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  if (op[i] != 0)
    op[i] = prime - op[i];
  STRIDED_LOOP_END;
}

__global__ void subInplace_(
    size_t degree,
    size_t log_degree,
    size_t batch,
    const uint64_t* primes,
    uint64_t* op1,
    const uint64_t* op2) {
  STRIDED_LOOP_START(batch * degree, i)
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  if (op1[i] >= op2[i]) {
    op1[i] -= op2[i];
  } else {
    op1[i] = prime - (op2[i] - op1[i]);
  }
  STRIDED_LOOP_END;
}

} // namespace fhe

namespace at::native {

static void iNTT_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  AT_DISPATCH_V2(
      kUInt64,
      "iNTT_cuda",
      AT_WRAP([&]() {
        dim3 gridDim(2048);
        dim3 blockDim(256);
        const int per_thread_ntt_size = 8;
        const int first_stage_radix_size = 256;
        const int second_radix_size = param_degree / first_stage_radix_size;
        const int pad = 4;
        const int per_thread_storage =
            blockDim.x * per_thread_ntt_size * sizeof(uint64_t);
        auto inverse_power_of_roots_div_two_ptr = reinterpret_cast<uint64_t*>(
            inverse_power_of_roots_div_two.data_ptr<uint64_t>());
        auto param_primes_ptr =
            reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
        auto inverse_scaled_power_of_roots_div_two_ptr =
            reinterpret_cast<uint64_t*>(
                inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());
        int gap = level - curr_limbs;

        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::Intt8PointPerThreadPhase2OoP<<<
            gridDim,
            blockDim,
            per_thread_storage,
            stream>>>(
            op_ptr,
            first_stage_radix_size,
            batch,
            param_degree,
            start_prime_idx,
            curr_limbs,
            gap,
            second_radix_size / per_thread_ntt_size,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr,
            op_ptr);
        fhe::Intt8PointPerThreadPhase1OoP<<<
            gridDim,
            (first_stage_radix_size / 8) * pad,
            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t),
            stream>>>(
            op_ptr,
            1,
            batch,
            param_degree,
            start_prime_idx,
            curr_limbs,
            gap,
            pad,
            first_stage_radix_size / 8,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr,
            op_ptr);
      }),
      kUInt64);
}

Tensor iNTT_cuda(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    int64_t curr_limbs,
    int64_t level) {
  auto res = op.clone();
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      start_prime_idx,
      batch,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  return res;
}

Tensor& iNTT_cuda_(
    Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    int64_t curr_limbs,
    int64_t level) {
  auto op_ptr = reinterpret_cast<uint64_t*>(op.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      start_prime_idx,
      batch,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  return op;
}

Tensor& iNTT_cuda_out(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    int64_t curr_limbs,
    int64_t level,
    Tensor& res) {
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      start_prime_idx,
      batch,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  return res;
}

static void NTT_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  dim3 gridDim(2048);
  dim3 blockDim(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = param_degree / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      blockDim.x * per_thread_ntt_size * sizeof(uint64_t);
  AT_DISPATCH_V2(
      kUInt64,
      "NTT_cuda",
      AT_WRAP([&]() {
        auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
            param_power_of_roots_shoup.data_ptr<uint64_t>());
        auto param_primes_ptr =
            reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
        auto param_power_of_roots_ptr = reinterpret_cast<uint64_t*>(
            param_power_of_roots.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::Ntt8PointPerThreadPhase1<<<
            gridDim,
            (first_stage_radix_size / 8) * pad,
            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t),
            stream>>>(
            op_ptr,
            1,
            batch,
            param_degree,
            start_prime_idx,
            pad,
            first_stage_radix_size / per_thread_ntt_size,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
        fhe::Ntt8PointPerThreadPhase2<<<
            gridDim,
            blockDim.x,
            per_thread_storage,
            stream>>>(
            op_ptr,
            first_stage_radix_size,
            batch,
            param_degree,
            start_prime_idx,
            second_radix_size / per_thread_ntt_size,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor NTT_cuda(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  auto res = op.clone();
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  NTT_impl(
      op_ptr,
      start_prime_idx,
      batch,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  return res;
}

Tensor& NTT_cuda_(
    Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  auto op_ptr = reinterpret_cast<uint64_t*>(op.data_ptr<uint64_t>());
  NTT_impl(
      op_ptr,
      start_prime_idx,
      batch,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  return op;
}

Tensor& NTT_cuda_out(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots,
    Tensor& res) {
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  NTT_impl(
      op_ptr,
      start_prime_idx,
      batch,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  return res;
}

static void NTT_except_some_range_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    int64_t excluded_range_start,
    int64_t excluded_range_size,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  auto excluded_range_end = excluded_range_start + excluded_range_size;
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = param_degree / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      block.x * per_thread_ntt_size * sizeof(uint64_t);
  AT_DISPATCH_V2(
      kUInt64,
      "NTT_except_some_range_impl",
      AT_WRAP([&]() {
        auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
            param_power_of_roots_shoup.data_ptr<uint64_t>());
        auto param_primes_ptr =
            reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
        auto param_power_of_roots_ptr = reinterpret_cast<uint64_t*>(
            param_power_of_roots.data_ptr<uint64_t>());
        int gap = level - curr_limbs;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::Ntt8PointPerThreadPhase1ExcludeSomeRange<<<
            grid,
            (first_stage_radix_size / 8) * pad,
            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t),
            stream>>>(
            op_ptr,
            1,
            batch,
            param_degree,
            start_prime_idx,
            excluded_range_start,
            excluded_range_end,
            curr_limbs,
            gap,
            pad,
            first_stage_radix_size / per_thread_ntt_size,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
        fhe::Ntt8PointPerThreadPhase2ExcludeSomeRange<<<
            grid,
            block.x,
            per_thread_storage,
            stream>>>(
            op_ptr,
            first_stage_radix_size,
            batch,
            param_degree,
            start_prime_idx,
            excluded_range_start,
            excluded_range_end,
            curr_limbs,
            gap,
            second_radix_size / per_thread_ntt_size,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void const_mult_batch_(
    uint64_t* op1_ptr,
    const Tensor& op2,
    const Tensor& op2_psinv,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t start_op1_idx,
    int64_t param_degree,
    uint64_t* res_ptr,
    const Tensor& primes) {
  AT_DISPATCH_V2(
      op2.scalar_type(),
      "const_mult_batch_",
      AT_WRAP([&]() {
        auto op2_ptr = reinterpret_cast<uint64_t*>(op2.data_ptr<uint64_t>());
        auto op2_psinv_ptr =
            reinterpret_cast<uint64_t*>(op2_psinv.data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = param_degree * batch / block_dim;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::const_mult_batch<<<grid_dim, block_dim, 0, stream>>>(
            (int)param_degree,
            primes_ptr,
            op1_ptr,
            op2_ptr,
            op2_psinv_ptr,
            (int)start_prime_idx,
            (int)batch,
            (int)start_op1_idx,
            res_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void modup_matmul_(
    uint64_t* ptr,
    int64_t beta_idx,
    uint64_t* to_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const int64_t param_alpha_,
    const int64_t param_degree_,
    at::TensorList prod_q_i_mod_q_j__,
    int64_t curr_limbs,
    int64_t level_) {
  const int unroll_factor = 4;
  const int begin_idx = (int)beta_idx * (int)param_alpha_;
  int start_length = ((begin_idx + param_alpha_) > curr_limbs)
      ? (curr_limbs - begin_idx)
      : param_alpha_;
  const int end_length = curr_limbs + param_alpha_ - start_length;
  int grid_dim{(int)param_degree_ * end_length / 256 / unroll_factor};
  int block_dim{256};
  const auto& prod_q_i_mod_q_j = prod_q_i_mod_q_j__[beta_idx];

  AT_DISPATCH_V2(
      kUInt64,
      "modup_matmul_",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        auto prod_q_i_mod_q_j_ptr =
            reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::modup_step_two_kernel<<<
            grid_dim,
            block_dim,
            prod_q_i_mod_q_j.size(-1) * sizeof(uint64_t),
            stream>>>(
            ptr,
            begin_idx,
            param_degree_,
            param_alpha_,
            curr_limbs,
            level_,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            prod_q_i_mod_q_j_ptr,
            prod_q_i_mod_q_j.size(-1),
            start_length,
            end_length,
            to_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void modup_impl_(
    uint64_t* from_ptr,
    uint64_t* to_ptr,
    int idx,
    int curr_limbs,
    int level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    const int64_t param_degree_,
    const int64_t param_alpha_,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  int num_moduli_after_modup = curr_limbs + param_alpha_;
  size_t begin_idx = idx * param_alpha_;
  size_t in_C_L_len = ((begin_idx + param_alpha_) > curr_limbs)
      ? (curr_limbs - begin_idx)
      : param_alpha_;

  auto hat_inverse_vec =
      hat_inverse_vec__[idx * param_alpha_ + (in_C_L_len - 1)];
  auto hat_inverse_vec_psinv =
      hat_inverse_vec_shoup__[idx * param_alpha_ + (in_C_L_len - 1)];

  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(
      to_ptr + (param_degree_ * begin_idx),
      from_ptr,
      8 * in_C_L_len * param_degree_,
      cudaMemcpyDeviceToDevice,
      stream);

  iNTT_impl(
      to_ptr,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      level,
      param_degree_,
      inverse_power_of_roots_div_two,
      param_primes__,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch_(
      to_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      begin_idx,
      in_C_L_len,
      begin_idx,
      param_degree_,
      to_ptr,
      param_primes__);

  modup_matmul_(
      to_ptr + param_degree_ * begin_idx,
      idx,
      to_ptr,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      param_alpha_,
      param_degree_,
      prod_q_i_mod_q_j__,
      curr_limbs,
      level);

  NTT_except_some_range_impl(
      to_ptr,
      0,
      num_moduli_after_modup,
      param_degree_,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      level,
      param_power_of_roots_shoup,
      param_primes__,
      param_power_of_roots);

  cudaMemcpyAsync(
      to_ptr + param_degree_ * begin_idx,
      from_ptr,
      8 * in_C_L_len * param_degree_,
      cudaMemcpyDeviceToDevice,
      stream);
}

static void modup(
    uint64_t* in_ptr,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    uint64_t* out_ptr) {
  int num_moduli_after_modup = curr_limbs + param_alpha_;
  for (int i = 0; i < beta; ++i) {
    modup_impl_(
        in_ptr + (param_alpha_ * param_degree_ * i),
        out_ptr + (num_moduli_after_modup * param_degree_) * i,
        i,
        curr_limbs,
        level,
        hat_inverse_vec__,
        hat_inverse_vec_shoup__,
        param_degree_,
        param_alpha_,
        param_primes__,
        param_barret_ratio__,
        param_barret_k__,
        prod_q_i_mod_q_j__,
        inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two,
        param_power_of_roots_shoup,
        param_power_of_roots);
  }
}

Tensor modup_cuda(
    const Tensor& out,
    const Tensor& in,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  auto res = out.clone();
  res.resize_({beta * (curr_limbs + param_alpha_) * param_degree_});
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  modup(
      in_ptr,
      curr_limbs,
      level,
      hat_inverse_vec__,
      hat_inverse_vec_shoup__,
      prod_q_i_mod_q_j__,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      beta,
      param_degree_,
      param_alpha_,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots,
      out_ptr);
  return res;
}
Tensor& modup_cuda_(
    Tensor& out,
    const Tensor& in,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  out.resize_({beta * (curr_limbs + param_alpha_) * param_degree_});
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  modup(
      in_ptr,
      curr_limbs,
      level,
      hat_inverse_vec__,
      hat_inverse_vec_shoup__,
      prod_q_i_mod_q_j__,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      beta,
      param_degree_,
      param_alpha_,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots,
      out_ptr);
  return out;
}

Tensor& modup_cuda_out(
    const Tensor& out,
    const Tensor& in,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    Tensor& res) {
  res.resize_({beta * (curr_limbs + param_alpha_) * param_degree_});
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  modup(
      in_ptr,
      curr_limbs,
      level,
      hat_inverse_vec__,
      hat_inverse_vec_shoup__,
      prod_q_i_mod_q_j__,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      beta,
      param_degree_,
      param_alpha_,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots,
      out_ptr);
  return res;
}

static void modup_core_impl_(
    uint64_t* from_ptr,
    uint64_t* to_ptr,
    int curr_limbs,
    int level,
    int idx,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    const int64_t param_degree_,
    const int64_t param_alpha_,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    at::TensorList prod_q_i_mod_q_j__) {
  int num_moduli_after_modup = curr_limbs + param_alpha_;
  size_t begin_idx = idx * param_alpha_;
  size_t in_C_L_len = ((begin_idx + param_alpha_) > curr_limbs)
      ? (curr_limbs - begin_idx)
      : param_alpha_;
  auto hat_inverse_vec =
      hat_inverse_vec__[idx * param_alpha_ + (in_C_L_len - 1)];
  auto hat_inverse_vec_psinv =
      hat_inverse_vec_shoup__[idx * param_alpha_ + (in_C_L_len - 1)];

  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(
      to_ptr + (param_degree_ * begin_idx),
      from_ptr,
      8 * in_C_L_len * param_degree_,
      cudaMemcpyDeviceToDevice,
      stream);

  const_mult_batch_(
      to_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      begin_idx,
      in_C_L_len,
      begin_idx,
      param_degree_,
      to_ptr,
      param_primes__);

  modup_matmul_(
      to_ptr + param_degree_ * begin_idx,
      idx,
      to_ptr,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      param_alpha_,
      param_degree_,
      prod_q_i_mod_q_j__,
      curr_limbs,
      level);

  cudaMemcpyAsync(
      to_ptr + param_degree_ * begin_idx,
      from_ptr,
      8 * in_C_L_len * param_degree_,
      cudaMemcpyDeviceToDevice,
      stream);
}

static void modup_core(
    uint64_t* in_ptr,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    uint64_t* out_ptr) {
  int num_moduli_after_modup = curr_limbs + param_alpha_;
  for (int i = 0; i < beta; ++i) {
    modup_core_impl_(
        in_ptr + (param_alpha_ * param_degree_ * i),
        out_ptr + (num_moduli_after_modup * param_degree_) * i,
        curr_limbs,
        level,
        i,
        hat_inverse_vec__,
        hat_inverse_vec_shoup__,
        param_degree_,
        param_alpha_,
        param_primes__,
        param_barret_ratio__,
        param_barret_k__,
        prod_q_i_mod_q_j__);
  }
}

Tensor modup_core_cuda(
    const Tensor& out,
    const Tensor& in,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_) {
  auto res = out.clone();
  res.resize_({beta * (curr_limbs + param_alpha_) * param_degree_});
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  modup_core(
      in_ptr,
      curr_limbs,
      level,
      hat_inverse_vec__,
      hat_inverse_vec_shoup__,
      prod_q_i_mod_q_j__,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      beta,
      param_degree_,
      param_alpha_,
      out_ptr);
  return res;
}

Tensor& modup_core_cuda_(
    Tensor& out,
    const Tensor& in,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_) {
  out.resize_({beta * (curr_limbs + param_alpha_) * param_degree_});
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  modup_core(
      in_ptr,
      curr_limbs,
      level,
      hat_inverse_vec__,
      hat_inverse_vec_shoup__,
      prod_q_i_mod_q_j__,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      beta,
      param_degree_,
      param_alpha_,
      out_ptr);
  return out;
}

Tensor& modup_core_cuda_out(
    const Tensor& out,
    const Tensor& in,
    int64_t curr_limbs,
    int64_t level,
    at::TensorList hat_inverse_vec__,
    at::TensorList hat_inverse_vec_shoup__,
    at::TensorList prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    Tensor& res) {
  res.resize_({beta * (curr_limbs + param_alpha_) * param_degree_});
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  modup_core(
      in_ptr,
      curr_limbs,
      level,
      hat_inverse_vec__,
      hat_inverse_vec_shoup__,
      prod_q_i_mod_q_j__,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      beta,
      param_degree_,
      param_alpha_,
      out_ptr);
  return res;
}

static void NegateInplace(
    uint64_t* op1,
    const int batch,
    const Tensor& primes,
    const int64_t param_degree,
    const int64_t param_log_degree) {
  AT_DISPATCH_V2(
      kUInt64,
      "NegateInplace",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * batch / block_dim;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::negateInplace_<<<grid_dim, block_dim, 0, stream>>>(
            param_degree, param_log_degree, batch, primes_ptr, op1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void SubInplace(
    uint64_t* op1,
    const uint64_t* op2,
    const int64_t batch,
    const int64_t param_degree,
    const int64_t param_log_degree,
    const Tensor& primes) {
  AT_DISPATCH_V2(
      kUInt64,
      "SubInplace",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * batch / block_dim;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::subInplace_<<<grid_dim, block_dim, 0, stream>>>(
            param_degree, param_log_degree, batch, primes_ptr, op1, op2);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void moddown_impl(
    uint64_t* from_ptr,
    const int64_t param_degree,
    const int64_t param_log_degree,
    const int64_t param_alpha_,
    const int64_t start_length,
    const int64_t end_length,
    const Tensor& primes,
    at::TensorList prod_q_i_mod_q_j_moddown,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    uint64_t* to_ptr) {
  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown[0];
  AT_DISPATCH_V2(
      kUInt64,
      "moddownImpl",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * end_length / block_dim;
        auto ptr = from_ptr + param_degree * end_length;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto param_barret_ratio_ptr = reinterpret_cast<uint64_t*>(
            param_barret_ratio.data_ptr<uint64_t>());
        auto param_barret_k_ptr =
            reinterpret_cast<uint64_t*>(param_barret_k.data_ptr<uint64_t>());
        auto prod_q_i_mod_q_j_ptr =
            reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::moddown_kernel<<<
            grid_dim,
            block_dim,
            prod_q_i_mod_q_j.size(-1) * sizeof(uint64_t),
            stream>>>(
            param_degree,
            primes_ptr,
            param_barret_ratio_ptr,
            param_barret_k_ptr,
            param_log_degree,
            ptr,
            prod_q_i_mod_q_j_ptr,
            start_length * end_length,
            param_alpha_,
            end_length,
            to_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void moddown_core_template(
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    Tensor& res) {
  const int start_length = level + alpha - curr_limbs; // tempK len
  const int end_length = curr_limbs;

  auto hat_inverse_vec = hat_inverse_vec_moddown.at(0);
  auto hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown.at(0);

  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  const_mult_batch_(
      from_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      level,
      alpha,
      curr_limbs,
      param_degree,
      from_ptr,
      param_primes);

  moddown_impl(
      from_ptr,
      param_degree,
      param_log_degree,
      alpha,
      start_length,
      end_length,
      param_primes,
      prod_q_i_mod_q_j_moddown,
      param_barret_ratio,
      param_barret_k,
      to_ptr);

  const auto& prod_inv = prod_inv_moddown.at(0);
  const auto& prod_inv_psinv = prod_inv_shoup_moddown.at(0);

  SubInplace(
      to_ptr,
      from_ptr,
      end_length,
      param_degree,
      param_log_degree,
      param_primes);

  NegateInplace(
      to_ptr, end_length, param_primes, param_degree, param_log_degree);

  const_mult_batch_(
      to_ptr,
      prod_inv,
      prod_inv_psinv,
      0,
      end_length,
      0,
      param_degree,
      to_ptr,
      param_primes);
}

Tensor moddown_core_cuda(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k) {
  auto res = to.clone();
  res.resize_({curr_limbs * param_degree});
  moddown_core_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      res);
  return res;
}

Tensor& moddown_core_cuda_(
    Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k) {
  to.resize_({curr_limbs * param_degree});
  moddown_core_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      to);
  return to;
}

Tensor& moddown_core_cuda_out(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    Tensor& res) {
  res.resize_({curr_limbs * param_degree});
  moddown_core_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      res);
  return res;
}

static void moddown_cuda_template(
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    Tensor& res) {
  const int start_length = alpha;
  const int end_length = curr_limbs;

  auto hat_inverse_vec = hat_inverse_vec_moddown[0];
  auto hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown[0];

  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  iNTT_impl(
      from_ptr,
      end_length,
      start_length,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch_(
      from_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      level,
      alpha,
      curr_limbs,
      param_degree,
      from_ptr,
      param_primes);

  moddown_impl(
      from_ptr,
      param_degree,
      param_log_degree,
      alpha,
      start_length,
      end_length,
      param_primes,
      prod_q_i_mod_q_j_moddown,
      param_barret_ratio,
      param_barret_k,
      to_ptr);

  NTT_impl(
      to_ptr,
      0,
      end_length,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  const auto& prod_inv = prod_inv_moddown.at(0);
  const auto& prod_inv_psinv = prod_inv_shoup_moddown.at(0);

  SubInplace(
      to_ptr,
      from_ptr,
      end_length,
      param_degree,
      param_log_degree,
      param_primes);

  NegateInplace(
      to_ptr, end_length, param_primes, param_degree, param_log_degree);

  const_mult_batch_(
      to_ptr,
      prod_inv,
      prod_inv_psinv,
      0,
      end_length,
      0,
      param_degree,
      to_ptr,
      param_primes);
}

Tensor moddown_cuda(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  auto res = to.clone();
  res.resize_({curr_limbs * param_degree});
  moddown_cuda_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      res);
  return res;
}

Tensor& moddown_cuda_(
    Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  to.resize_({curr_limbs * param_degree});
  moddown_cuda_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      to);
  return to;
}
Tensor& moddown_cuda_out(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    at::TensorList hat_inverse_vec_moddown,
    at::TensorList hat_inverse_vec_shoup_moddown,
    at::TensorList prod_q_i_mod_q_j_moddown,
    at::TensorList prod_inv_moddown,
    at::TensorList prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    Tensor& res) {
  res.resize_({curr_limbs * param_degree});
  moddown_cuda_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      res);
  return res;
}

} // namespace at::native