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

#include "ATen/native/fhe/cuda/KeySwitch.h"
#include "ATen/native/fhe/cuda/NttImpl.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__device__ uint128_t4 accumulate_in_modup(
    const uint64_t* ptr,
    const int N,
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
        : "l"(ptr + i * N + degree_idx));

    out.x = mult_64_64_128(op1_x, op2);
    inplace_add_128_128(out.x, accum.x);
    out.y = mult_64_64_128(op1_y, op2);
    inplace_add_128_128(out.y, accum.y);
    asm("{\n\t"
        "ld.global.v2.u64 {%0, %1}, [%2];\n\t"
        "}"
        : "=l"(op1_z), "=l"(op1_w)
        : "l"(ptr + i * N + degree_idx + 2));
    out.z = mult_64_64_128(op1_z, op2);
    inplace_add_128_128(out.z, accum.z);
    out.w = mult_64_64_128(op1_w, op2);
    inplace_add_128_128(out.w, accum.w);
  }
  return accum;
}

__global__ void modup_step_two_simple(
    const uint64_t* ptr_after_intt,
    const uint64_t* ptr_before_intt,
    const int in_prime_idx,
    const int N,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
    const uint64_t end_length,
    uint64_t* to) {
  STRIDED_LOOP_START(N * end_length, i);
  const int out_prime_idx = i / N;
  const int degree_idx = i % N;
  const auto barret_ratio = barrett_ratios[out_prime_idx];
  const auto barret_k = barrett_ks[out_prime_idx];
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

__global__ void const_mult_batch_kernel(
    uint64_t* op1,
    const uint64_t* op2,
    const uint64_t* op2_psinv,
    const uint64_t* primes,
    const size_t N,
    const int start_prime_idx,
    const int batch,
    const int start_op1_idx,
    const int start_op2_idx,
    uint64_t* to) {
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

__global__ void modup_step_two_kernel(
    const uint64_t* ptr,
    const int begin_idx,
    const int N,
    const int alpha,
    const int curr_limbs,
    const int L,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
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
  STRIDED_LOOP_START((N * end_length + unroll_number - 1) / unroll_number, i);
  const int degree_idx = unroll_number * (i / end_length);
  const int hat_mod_end_idx = i % end_length;
  const int out_idx =
      hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);
  uint128_t4 accum = accumulate_in_modup(
      ptr, N, s_hat_mod_end, alpha, degree_idx, hat_mod_end_idx);
  int gap = L - curr_limbs;
  int prime_idx = out_idx +
      (((out_idx >= 0 && out_idx < begin_idx) ||
        (out_idx >= (begin_idx + start_length) && out_idx < curr_limbs))
           ? 0
           : gap);
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barrett_ratios[prime_idx];
  const auto barret_k = barrett_ks[prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_idx * N + degree_idx),
        "l"(out),
        "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_idx * N + degree_idx + 2),
        "l"(out),
        "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void moddown_kernel(
    const uint64_t* ptr,
    const int64_t N,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks,
    const uint64_t* hat_mod_end,
    const int hat_mod_end_size,
    const uint64_t start_length, // it should be the size of the Auxiliary CRT
                                 // basis {P} = {p_1,...,p_k}
    const uint64_t end_length, // it should be curr_limbs
    uint64_t* to) {
  constexpr const int unroll_number = 4;
  extern __shared__ uint64_t s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  STRIDED_LOOP_START((N * end_length + unroll_number - 1) / unroll_number, i);
  const int degree_idx = unroll_number * (i / end_length);
  const int out_prime_idx = i % end_length;
  uint128_t4 accum = accumulate_in_modup(
      ptr, N, s_hat_mod_end, start_length, degree_idx, out_prime_idx);
  const auto prime = primes[out_prime_idx];
  const auto barret_ratio = barret_ratios[out_prime_idx];
  const auto barret_k = barret_ks[out_prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * N + degree_idx),
        "l"(out),
        "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * N + degree_idx + 2),
        "l"(out),
        "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void negate_inplace_kernel(
    uint64_t* op,
    const size_t N,
    const size_t log_degree,
    const size_t batch,
    const uint64_t* primes) {
  STRIDED_LOOP_START(batch * N, i);
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  if (op[i] != 0)
    op[i] = prime - op[i];
  STRIDED_LOOP_END;
}

__global__ void sub_inplace_kernel(
    const uint64_t* op2,
    size_t N,
    size_t batch,
    const uint64_t* primes,
    uint64_t* op1) {
  STRIDED_LOOP_START(batch * N, i)
  const int prime_idx = i / N;
  const uint64_t prime = primes[prime_idx];
  if (op1[i] >= op2[i]) {
    op1[i] -= op2[i];
  } else {
    op1[i] = prime - (op2[i] - op1[i]);
  }
  STRIDED_LOOP_END;
}

__global__ void vec_add_mod_batch_kernel(
    const uint64_t* op1,
    const uint64_t* op2,
    const int N,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks,
    const uint64_t batch,
    uint64_t* to) {
  STRIDED_LOOP_START((N * batch), i);
  const int out_prime_idx = i / N;
  const auto prime = primes[out_prime_idx];
  const auto barret_ratio = barret_ratios[out_prime_idx];
  const auto barret_k = barret_ks[out_prime_idx];
  barret_reduction_64_64(op1[i] + op2[i], to[i], prime, barret_ratio, barret_k);

  STRIDED_LOOP_END;
}

__global__ void vec_mod_batch_kernel(
    const uint64_t* op1,
    const int N,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks,
    const uint64_t batch,
    uint64_t* to) {
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
    const uint64_t* ptr,
    const size_t N,
    const size_t batch,
    const size_t old_prime_idx,
    const uint64_t* primes,
    uint64_t* to) {
  STRIDED_LOOP_START(batch * N, i)
  auto old_modulus_by_two = primes[old_prime_idx] >> 1;
  auto old_modulus = primes[old_prime_idx];
  auto new_modulus_idx = i / N;
  auto new_modulus = primes[new_modulus_idx];
  auto diff = (old_modulus > new_modulus)
      ? (new_modulus- (old_modulus%new_modulus)) //fixme: remove the `%`, note that in mod_raise case, the quotient of om/nm may be >=2
      : (new_modulus - old_modulus);
  int input_idx = i % N;
  auto tmp = (ptr[input_idx] > old_modulus_by_two) ? diff : 0;

  if (new_modulus >= old_modulus) {
    to[i] = tmp + ptr[input_idx];
  } else { // old_modulus > new_modulus
    // deprecated
    //     if (ptr[input_idx] >= tmp) {
    //       to[i] = ptr[input_idx] - tmp;
    //     } else {
    //       to[i] = new_modulus - (tmp - ptr[input_idx]);
    //     }
    to[i] = tmp + ptr[input_idx];
    if (to[i]>=new_modulus)
        to[i] = to[i] % new_modulus; // fixme: note that quotient>=1, can not replaced with sub trivially
  }
  STRIDED_LOOP_END;
}

} // namespace fhe

namespace at::native {

static void NTT_except_some_range_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t N,
    int64_t excluded_range_start,
    int64_t excluded_range_size,
    int64_t curr_limbs,
    int64_t L,
    const Tensor& power_of_roots_shoup,
    const Tensor& primes,
    const Tensor& power_of_roots) {
  auto excluded_range_end = excluded_range_start + excluded_range_size;
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = N / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      block.x * per_thread_ntt_size * sizeof(uint64_t);
  AT_DISPATCH_V2(
      kUInt64,
      "NTT_except_some_range_impl",
      AT_WRAP([&]() {
        auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
            power_of_roots_shoup.data_ptr<uint64_t>());
        auto param_primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto param_power_of_roots_ptr =
            reinterpret_cast<uint64_t*>(power_of_roots.data_ptr<uint64_t>());
        int gap = L - curr_limbs;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::Ntt8PointPerThreadPhase1ExcludeSomeRange<<<
            grid,
            (first_stage_radix_size / 8) * pad,
            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t),
            stream>>>(
            op_ptr,
            1,
            batch,
            N,
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
            N,
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

void const_mult_batch(
    uint64_t* op1_ptr,
    const Tensor& op2,
    const Tensor& op2_psinv,
    const Tensor& primes,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t start_op1_idx,
    int64_t start_op2_idx,
    int64_t N,
    uint64_t* out_ptr) {
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
            op1_ptr,
            op2_ptr,
            op2_psinv_ptr,
            primes_ptr,
            (int)N,
            (int)start_prime_idx,
            (int)batch,
            (int)start_op1_idx,
            (int)start_op2_idx,
            out_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void modup_matmul(
    uint64_t* from_ptr,
    int64_t beta_idx,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const int64_t alpha,
    const int64_t N,
    const Tensor& prod_q_i_mod_q_js,
    int64_t curr_limbs,
    int64_t L,
    uint64_t* to_ptr) {
  const int unroll_factor = 4;
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int begin_idx = (int)beta_idx * (int)alpha;
  int start_length =
      ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;
  const int end_length = curr_limbs + sizeP - start_length;
  int grid_dim{(int)N * end_length / 256 / unroll_factor};
  int block_dim{256};
  const auto& prod_q_i_mod_q_j = prod_q_i_mod_q_js[beta_idx];

  AT_DISPATCH_V2(
      kUInt64,
      "modup_matmul",
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
            from_ptr,
            begin_idx,
            N,
            alpha,
            curr_limbs,
            L,
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

static void modup_impl(
    uint64_t* from_ptr,
    int idx,
    int curr_limbs,
    int L,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const int64_t N,
    const int64_t alpha,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    uint64_t* to_ptr) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  int num_moduli_after_modup = curr_limbs + sizeP;
  size_t begin_idx = idx * alpha;
  size_t in_C_L_len =
      ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;

  auto hat_inverse_vec = hat_inverse_vecs[idx * alpha + (in_C_L_len - 1)];
  auto hat_inverse_vec_psinv =
      hat_inverse_vec_shoups[idx * alpha + (in_C_L_len - 1)];

  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(
      to_ptr + (N * begin_idx),
      from_ptr,
      8 * in_C_L_len * N,
      cudaMemcpyDeviceToDevice,
      stream);

  iNTT_impl(
      to_ptr,
      to_ptr,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      L,
      N,
      inverse_power_of_roots_div_two,
      primes,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch(
      to_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      primes,
      begin_idx,
      in_C_L_len,
      begin_idx,
      0,
      N,
      to_ptr);

  modup_matmul(
      to_ptr + N * begin_idx,
      idx,
      primes,
      barret_ratio,
      barret_k,
      alpha,
      N,
      prod_q_i_mod_q_js,
      curr_limbs,
      L,
      to_ptr);

  NTT_except_some_range_impl(
      to_ptr,
      0,
      num_moduli_after_modup,
      N,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      L,
      power_of_roots_shoup,
      primes,
      power_of_roots);

  cudaMemcpyAsync(
      to_ptr + N * begin_idx,
      from_ptr,
      8 * in_C_L_len * N,
      cudaMemcpyDeviceToDevice,
      stream);
}

static void modup_cuda_template(
    uint64_t* in_ptr,
    int64_t curr_limbs,
    int64_t L,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    uint64_t* out_ptr) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  int num_moduli_after_modup = curr_limbs + sizeP;
  for (int i = 0; i < beta; ++i) {
    modup_impl(
        in_ptr + (alpha * N * i),
        i,
        curr_limbs,
        L,
        hat_inverse_vecs,
        hat_inverse_vec_shoups,
        N,
        alpha,
        primes,
        barret_ratio,
        barret_k,
        prod_q_i_mod_q_js,
        inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two,
        power_of_roots_shoup,
        power_of_roots,
        out_ptr + (num_moduli_after_modup * N) * i);
  }
}

static void negate_inplace(
    uint64_t* op1,
    const int batch,
    const Tensor& primes,
    const int64_t N,
    const int64_t log_degree) {
  AT_DISPATCH_V2(
      kUInt64,
      "negate_inplace",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = N * batch / block_dim;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::negate_inplace_kernel<<<grid_dim, block_dim, 0, stream>>>(
            op1, N, log_degree, batch, primes_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

void sub_inplace(
    const uint64_t* from_ptr,
    const int64_t batch,
    const int64_t N,
    const Tensor& primes,
    uint64_t* to_ptr) {
  AT_DISPATCH_V2(
      kUInt64,
      "sub_inplace",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = N * batch / block_dim;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::sub_inplace_kernel<<<grid_dim, block_dim, 0, stream>>>(
            from_ptr, N, batch, primes_ptr, to_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void moddown_impl(
    uint64_t* from_ptr,
    const int64_t N,
    const int64_t alpha,
    const int64_t start_length,
    const int64_t end_length,
    const Tensor& primes,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    uint64_t* to_ptr) {
  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown[0];
  AT_DISPATCH_V2(
      kUInt64,
      "moddownImpl",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = N * end_length / block_dim;
        auto ptr = from_ptr + N * end_length;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto param_barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto param_barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        auto prod_q_i_mod_q_j_ptr =
            reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::moddown_kernel<<<
            grid_dim,
            block_dim,
            prod_q_i_mod_q_j.size(-1) * sizeof(uint64_t),
            stream>>>(
            ptr,
            N,
            primes_ptr,
            param_barret_ratio_ptr,
            param_barret_k_ptr,
            prod_q_i_mod_q_j_ptr,
            start_length * end_length,
            alpha,
            end_length,
            to_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void moddown_cuda_template(
    const Tensor& from,
    int64_t curr_limbs,
    int64_t L,
    int64_t alpha,
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
    Tensor& res) {
  const int start_length = alpha;
  const int end_length = curr_limbs;

  auto hat_inverse_vec = hat_inverse_vec_moddown[0];
  auto hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown[0];

  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  iNTT_impl(
      from_ptr,
      from_ptr,
      end_length,
      start_length,
      curr_limbs,
      L,
      N,
      inverse_power_of_roots_div_two,
      primes,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch(
      from_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      primes,
      L,
      alpha,
      curr_limbs,
      0,
      N,
      from_ptr);

  moddown_impl(
      from_ptr,
      N,
      alpha,
      start_length,
      end_length,
      primes,
      prod_q_i_mod_q_j_moddown,
      barret_ratio,
      barret_k,
      to_ptr);

  NTT_impl(
      to_ptr,
      to_ptr,
      0,
      end_length,
      N,
      power_of_roots_shoup,
      primes,
      power_of_roots);

  const auto& prod_inv = prod_inv_moddown[0];
  const auto& prod_inv_psinv = prod_inv_shoup_moddown[0];

  sub_inplace(from_ptr, end_length, N, primes, to_ptr);

  negate_inplace(to_ptr, end_length, primes, N, log_degree);

  const_mult_batch(
      to_ptr, prod_inv, prod_inv_psinv, primes, 0, end_length, 0, 0, N, to_ptr);
}

Tensor modup_cuda(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out = at::empty(beta * (curr_limbs + sizeP) * N, in.options());
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  modup_cuda_template(
      in_ptr,
      curr_limbs,
      L,
      hat_inverse_vecs,
      hat_inverse_vec_shoups,
      prod_q_i_mod_q_js,
      primes,
      barret_ratio,
      barret_k,
      beta,
      N,
      alpha,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      power_of_roots_shoup,
      power_of_roots,
      out_ptr);
  return out;
}

Tensor moddown_cuda(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    int64_t alpha,
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
  auto out = at::empty(curr_limbs * N, in.options());
  moddown_cuda_template(
      in.clone(),
      curr_limbs,
      L,
      alpha,
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
      out);
  return out;
}

void vec_add_mod_batch(
    uint64_t* in1_ptr,
    uint64_t* in2_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t batch,
    int64_t N,
    uint64_t* out_ptr) {
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
            in1_ptr,
            in2_ptr,
            (int)N,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            (int)batch,
            out_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

void vec_mod_batch(
    uint64_t* in_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t batch,
    int64_t N,
    uint64_t* out_ptr) {
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
            in_ptr,
            (int)N,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            (int)batch,
            out_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

void switch_modulus(
    uint64_t* in_ptr,
    uint64_t* out_ptr,
    const Tensor& primes,
    int64_t old_prime_index,
    int64_t batch,
    int64_t N) {
  AT_DISPATCH_V2(
      primes.scalar_type(),
      "switch_modulus",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = N * batch / block_dim;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::switch_modulus_kernel<<<grid_dim, block_dim, 0, stream>>>(
            in_ptr, (int)N, batch, old_prime_index, primes_ptr, out_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

} // namespace at::native