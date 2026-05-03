#pragma once
#include <ATen/core/Tensor.h>
#include <cstdint>

namespace at::native {
void iNTT_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    const size_t num_batch,
    const size_t N,
    const size_t L_OUT,
    const size_t L_IN,
    const size_t num_cv,
    const size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* inverse_power_of_roots_div_two,
    const uint64_t* inverse_scaled_power_of_roots_div_two);

void NTT_impl(
    uint64_t* inout_ptr,
    const size_t num_batch,
    const size_t N,
    const size_t L,
    const size_t num_cv,
    const size_t num_cipher,
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

} // namespace at::native
