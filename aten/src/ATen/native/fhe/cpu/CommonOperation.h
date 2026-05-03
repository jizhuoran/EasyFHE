#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/fhe/cpu/arithmetic.h>
#include <cstddef>
#include <cstdint>

namespace at::native {

// 4D kernels API aligned with CUDA implementation.
void iNTT_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t num_batch,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* inverse_power_of_roots_div_two,
    const uint64_t* inverse_scaled_power_of_roots_div_two);

void NTT_impl(
    uint64_t* inout_ptr,
    size_t num_batch,
    size_t N,
    size_t L,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* param_power_of_roots_shoup,
    const uint64_t* param_power_of_roots);

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
    const Tensor& switch_modulus_map);

void const_mult_batch(
    uint64_t* res_ptr,
    const uint64_t* op1_ptr,
    const uint64_t* op2_ptr,
    const uint64_t* op2_psinv_ptr,
    size_t batch,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* primes_ptr);

// Legacy wrappers kept for old callers in CPU code paths.
void iNTT_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
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
    int degree,
    const uint64_t* hat_mod_end,
    int start_length,
    int degree_idx,
    int hat_mod_end_idx);

} // namespace fhe
