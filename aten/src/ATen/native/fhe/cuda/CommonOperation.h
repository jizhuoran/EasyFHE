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

void iNTT_scaled_impl(
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
    const uint64_t* inverse_scaled_power_of_roots_div_two,
    const uint64_t* scalars,
    const uint64_t* scalar_shoups);

void iNTT_modup_scaled_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    const size_t curr_limbs,
    const size_t N,
    const size_t L_OUT,
    const size_t L_IN,
    const size_t num_cv,
    const size_t num_cipher,
    const size_t alpha,
    const uint64_t* param_primes,
    const uint64_t* inverse_power_of_roots_div_two,
    const uint64_t* inverse_scaled_power_of_roots_div_two,
    const uint64_t* scalars,
    const uint64_t* scalar_shoups,
    const size_t scalar_stride);

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

void NTT_modup_masked_impl(
    uint64_t* inout_ptr,
    const size_t num_batch,
    const size_t curr_limbs,
    const size_t N,
    const size_t L,
    const size_t begin_idx,
    const size_t group_size,
    const size_t L_OUT,
    const size_t num_cv,
    const size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* param_power_of_roots_shoup,
    const uint64_t* param_power_of_roots);

void NTT_modup_all_masked_impl(
    uint64_t* inout_ptr,
    const size_t beta,
    const size_t curr_limbs,
    const size_t N,
    const size_t L,
    const size_t alpha,
    const size_t num_moduli_after_modup,
    const size_t L_OUT,
    const size_t num_cv,
    const size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* param_power_of_roots_shoup,
    const uint64_t* param_power_of_roots);

void modup_step_two_ntt_all_impl(
    uint64_t* out_ptr,
    const uint64_t* in_ptr,
    const size_t beta,
    const size_t curr_limbs,
    const size_t N,
    const size_t L,
    const size_t alpha,
    const size_t num_moduli_after_modup,
    const size_t L_OUT,
    const size_t L_IN,
    const size_t num_cv,
    const size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
    const uint64_t* prod_q_i_mod_q_js,
    const size_t prod_beta_stride,
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
    const Tensor& inverse_scaled_power_of_roots_div_two);

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
    const Tensor& inverse_scaled_power_of_roots_div_two);

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
    const Tensor& inverse_scaled_power_of_roots_div_two);

Tensor innerproduct_cuda(
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace);

Tensor moddown_cuda(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    int64_t sizeP,
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
    const Tensor& inverse_scaled_power_of_roots_div_two);

Tensor rescale_one_level_cuda(
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    int64_t old_primes,
    const Tensor& param_primes,
    const Tensor& switch_modulus_map,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup);

} // namespace at::native
