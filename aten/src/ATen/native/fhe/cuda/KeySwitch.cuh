
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <crt/device_functions.h>
#include "ATen/native/fhe/cuda/Utils.cuh"

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
    uint64_t* op1,
    const uint64_t* op2,
    const uint64_t* op2_psinv,
    const int start_prime_idx,
    const int batch,
    const int start_op1_idx,
    const int start_op2_idx,
    uint64_t* to) {
  STRIDED_LOOP_START(degree * batch, i);
  const int op2_idx = start_op2_idx + i / degree;
  const int prime_idx = i / degree + start_prime_idx;
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
  int gap = level - curr_limbs;
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
    size_t batch,
    const uint64_t* primes,
    uint64_t* op1,
    const uint64_t* op2) {
  STRIDED_LOOP_START(batch * degree, i)
  const int prime_idx = i / degree;
  const uint64_t prime = primes[prime_idx];
  if (op1[i] >= op2[i]) {
    op1[i] -= op2[i];
  } else {
    op1[i] = prime - (op2[i] - op1[i]);
  }
  STRIDED_LOOP_END;
}

__global__ void vec_add_mod_batch_(
    int degree_,
    uint64_t* d_primes,
    uint64_t* d_barret_ratio,
    uint64_t* d_barret_k,
    const uint64_t* op1,
    const uint64_t* op2,
    const uint64_t batch,
    uint64_t* to) {
  STRIDED_LOOP_START((degree_ * batch), i);
  // const int degree_idx = unroll_number * (i / end_length);
  const int out_prime_idx = i / degree_;
  // const int in_idx = i % degree_;
  const auto prime = d_primes[out_prime_idx];
  const auto barret_ratio = d_barret_ratio[out_prime_idx];
  const auto barret_k = d_barret_k[out_prime_idx];
  barret_reduction_64_64(op1[i] + op2[i], to[i], prime, barret_ratio, barret_k);

  STRIDED_LOOP_END;
}

__global__ void vec_mod_batch_(
    int degree_,
    uint64_t* d_primes,
    uint64_t* d_barret_ratio,
    uint64_t* d_barret_k,
    const uint64_t* op1,
    const uint64_t batch,
    uint64_t* to) {
  STRIDED_LOOP_START((degree_ * batch), i);
  const int out_prime_idx = i / degree_;
  const int op1_idx = i % degree_;
  const auto prime = d_primes[out_prime_idx];
  const auto barret_ratio = d_barret_ratio[out_prime_idx];
  const auto barret_k = d_barret_k[out_prime_idx];
  barret_reduction_64_64(op1[op1_idx], to[i], prime, barret_ratio, barret_k);

  STRIDED_LOOP_END;
}

__global__ void switch_modulus_(
    size_t degree,
    size_t batch,
    const size_t old_prime_idx,
    const uint64_t* primes,
    const uint64_t* ptr,
    uint64_t* to) {
  STRIDED_LOOP_START(batch * degree, i)
  auto old_modulus_by_two = primes[old_prime_idx] >> 1;
  auto old_modulus = primes[old_prime_idx];
  auto new_modulus_idx = i / degree;
  auto diff = (old_modulus > primes[new_modulus_idx])
      ? (old_modulus - primes[new_modulus_idx])
      : (primes[new_modulus_idx] - old_modulus);
  int input_idx = i % degree;
  auto tmp = (ptr[input_idx] > old_modulus_by_two) ? diff : 0;
  if (primes[new_modulus_idx] >= old_modulus) {
    to[i] = tmp + ptr[input_idx];
  } else {
    if (ptr[input_idx] >= tmp) {
      to[i] = ptr[input_idx] - tmp;
    } else {
      to[i] = primes[new_modulus_idx] - (tmp - ptr[input_idx]);
    }
    // to[i] = to[i] % primes[new_modulus_idx];
  }
  STRIDED_LOOP_END;
}

} // namespace fhe
