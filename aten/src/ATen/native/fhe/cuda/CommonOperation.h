#pragma once
#include <ATen/core/Tensor.h>
#include <cstdint>

namespace at::native {
void iNTT_impl(
    uint64_t* in_ptr,
    uint64_t* out_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two);

void NTT_impl(
    const uint64_t* in_ptr,
    uint64_t* out_ptr,
    int64_t batch,
    int64_t param_degree,
    const uint64_t* param_power_of_roots_shoup,
    const uint64_t* param_primes,
    const uint64_t* param_power_of_roots);

void NTT_except_some_range_impl(
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
    const Tensor& power_of_roots);

void switch_modulus(
    uint64_t* res_ptr,
    uint64_t* ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t old_prime_index,
    int64_t batch,
    int64_t degree);

void const_mult_batch(
    uint64_t* res_ptr,
    const uint64_t* op1_ptr,
    const uint64_t* op2_ptr,
    const uint64_t* op2_psinv_ptr,
    const uint64_t* primes_ptr,
    int64_t batch,
    int64_t param_degree);

} // namespace at::native