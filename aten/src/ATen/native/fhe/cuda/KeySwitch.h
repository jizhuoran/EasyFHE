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
    uint64_t* in_ptr,
    uint64_t* out_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots);

void switch_modulus(
    uint64_t* res_ptr,
    uint64_t* ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t old_prime_index,
    int64_t batch,
    int64_t degree);

void vec_mod_batch(
    uint64_t* res_ptr,
    uint64_t* op1_ptr,
    const Tensor& primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    int64_t batch,
    int64_t degree);

void vec_add_mod_batch(
    uint64_t* res_ptr,
    uint64_t* op1_ptr,
    uint64_t* op2_ptr,
    const Tensor& primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    int64_t batch,
    int64_t degree);

void const_mult_batch(
    uint64_t* res_ptr,
    uint64_t* op1_ptr,
    const Tensor& op2,
    const Tensor& op2_psinv,
    const Tensor& primes,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t start_op1_idx,
    int64_t start_op2_idx,
    int64_t param_degree);

void sub_inplace(
    uint64_t* to_ptr,
    const uint64_t* from_ptr,
    const int64_t batch,
    const int64_t param_degree,
    const Tensor& primes);

} // namespace at::native