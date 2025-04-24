#pragma once
#include <cstdint>
#include <ATen/core/Tensor.h>

namespace at::native {
void iNTT_impl(
    uint64_t* op_ptr,
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
    uint64_t* out_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
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

void vec_add_mod_batch(
    uint64_t* op1_ptr,
    uint64_t* op2_ptr,
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

void SubInplace(
    uint64_t* op1,
    const uint64_t* op2,
    const int64_t batch,
    const int64_t param_degree,
    const Tensor& primes);

} // namespace at::native
