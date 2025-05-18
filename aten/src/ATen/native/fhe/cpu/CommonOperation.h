#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/fhe/cpu/arithmetic.h>
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
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots);

void NTT_except_some_range_impl(
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
    const Tensor& param_power_of_roots);

void switch_modulus(
    uint64_t* ptr,
    uint64_t* res_ptr,
    const Tensor& primes,
    int64_t old_prime_index,
    int64_t batch,
    int64_t degree);

void vec_mod_batch(
    uint64_t* op1_ptr,
    const Tensor& primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    int64_t batch,
    int64_t degree,
    uint64_t* res_ptr);

void const_mult_batch_(
    uint64_t* op1_ptr,
    const Tensor& op2,
    const Tensor& op2_psinv,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t start_op1_idx,
    int64_t start_op2_idx,
    int64_t param_degree,
    uint64_t* res_ptr,
    const Tensor& primes);

} // namespace at::native

namespace fhe {
__uint128_t accumulate_in_modup(
    const uint64_t* ptr,
    const int degree,
    const uint64_t* hat_mod_end,
    const int start_length, // sizeP
    const int degree_idx,
    const int hat_mod_end_idx);
}