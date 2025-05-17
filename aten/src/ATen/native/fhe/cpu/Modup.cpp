#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <omp.h>
#include <iostream>
#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/NttImpl.h"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
namespace fhe {

void modup_step_two_kernel(
    const uint64_t* ptr,
    const int begin_idx,
    const int degree, // ringDim
    const int alpha,
    const int curr_limbs,
    const int level,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_Ks,
    const uint64_t* hat_mod_end,
    const int hat_mod_end_size,
    const uint64_t start_length, // sizeP
    const uint64_t end_length, // sizeQ
    uint64_t* to) {
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);

#pragma omp parallel for num_threads(max_threads)
  for (int hat_mod_end_idx = 0; hat_mod_end_idx < end_length;
       hat_mod_end_idx++) {
    const int out_idx =
        hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);
    int gap = level - curr_limbs;
    int prime_idx = out_idx +
        (((out_idx >= 0 && out_idx < begin_idx) ||
          (out_idx >= (begin_idx + start_length) && out_idx < curr_limbs))
             ? 0
             : gap);
    const auto prime = primes[prime_idx];
    const auto barret_ratio = barrett_ratios[prime_idx];
    const auto barret_k = barrett_Ks[prime_idx];
    for (int degree_idx = 0; degree_idx < degree; degree_idx++) {
      __uint128_t accum = accumulate_in_modup(
          ptr, degree, hat_mod_end, alpha, degree_idx, hat_mod_end_idx);

      // First store operation
      uint64_t out1 =
          barret_reduction_128_64(accum, prime, barret_ratio, barret_k);
      to[out_idx * degree + degree_idx] = out1;
    }
  }
}
} // namespace fhe

namespace at::native {

static void modup_matmul_(
    uint64_t* ptr,
    int64_t beta_idx,
    uint64_t* to_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const int64_t param_alpha_,
    const int64_t param_degree_,
    const Tensor& prod_q_i_mod_q_j__,
    int64_t curr_limbs,
    int64_t level_) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - level_;
  const int unroll_factor = 1;
  const int begin_idx = (int)beta_idx * (int)param_alpha_;
  int start_length = ((begin_idx + param_alpha_) > curr_limbs)
      ? (curr_limbs - begin_idx)
      : param_alpha_;
  const int end_length = curr_limbs + sizeP - start_length;
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
        fhe::modup_step_two_kernel(
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
      }),
      kUInt64);
}

static void modup_impl_(
    uint64_t* from_ptr,
    uint64_t* to_ptr,
    int idx,
    int curr_limbs,
    int level, // fixme: change all these var `level` into `total_limbs` or `L` for clarity?
    const Tensor& hat_inverse_vec__,
    const Tensor& hat_inverse_vec_shoup__,
    const int64_t param_degree_,
    const int64_t param_alpha_,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    const Tensor& prod_q_i_mod_q_j__,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  int64_t sizeQP = param_primes__.numel();
  int64_t sizeP = sizeQP - level;
  int num_moduli_after_modup = curr_limbs + sizeP;
  size_t begin_idx = idx * param_alpha_;
  size_t in_C_L_len = ((begin_idx + param_alpha_) > curr_limbs)
      ? (curr_limbs - begin_idx)
      : param_alpha_;

  auto hat_inverse_vec =
      hat_inverse_vec__[idx * param_alpha_ + (in_C_L_len - 1)];
  auto hat_inverse_vec_psinv =
      hat_inverse_vec_shoup__[idx * param_alpha_ + (in_C_L_len - 1)];
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);

  memcpy(
      to_ptr + (param_degree_ * begin_idx),
      from_ptr,
      8 * in_C_L_len * param_degree_);

  iNTT_impl(
      to_ptr,
      to_ptr,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      level,
      param_degree_,
      inverse_power_of_roots_div_two,
      param_primes__,
      inverse_scaled_power_of_roots_div_two);
  auto op2_ptr =
      reinterpret_cast<uint64_t*>(hat_inverse_vec.data_ptr<uint64_t>());
  auto op2_psinv_ptr =
      reinterpret_cast<uint64_t*>(hat_inverse_vec_psinv.data_ptr<uint64_t>());
  auto primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes__.data_ptr<uint64_t>());
#pragma omp parallel for num_threads(max_threads)
  for (int i = 0; i < param_degree_ * in_C_L_len; i++) {
    const int op2_idx = 0 + i / param_degree_;
    const int prime_idx = i / param_degree_ + begin_idx;
    const auto prime = primes_ptr[prime_idx];
    uint64_t out = fhe::mul_and_reduce_shoup(
        to_ptr[begin_idx * param_degree_ + i],
        op2_ptr[op2_idx],
        op2_psinv_ptr[op2_idx],
        prime);
    if (out >= prime)
      out -= prime;
    to_ptr[begin_idx * param_degree_ + i] = out;
  }
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

  memcpy(
      to_ptr + (param_degree_ * begin_idx),
      from_ptr,
      8 * in_C_L_len * param_degree_);
}

static void modup(
    uint64_t* in_ptr,
    uint64_t* out_ptr,
    int64_t curr_limbs,
    int64_t level, // fixme: change all these var `level` into `total_limbs` or `L` for clarity?
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    const Tensor& hat_inverse_vec__,
    const Tensor& hat_inverse_vec_shoup__,
    const Tensor& prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  int64_t sizeQP = param_primes__.numel();
  int64_t sizeP = sizeQP - level;
  int num_moduli_after_modup = curr_limbs + sizeP;
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

Tensor modup_cpu(
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
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out = at::empty(beta * (curr_limbs + sizeP) * N, in.options());
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  modup(
      in_ptr,
      out_ptr,
      curr_limbs,
      L,
      beta,
      N,
      alpha,
      hat_inverse_vecs,
      hat_inverse_vec_shoups,
      prod_q_i_mod_q_js,
      primes,
      barret_ratio,
      barret_k,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      power_of_roots_shoup,
      power_of_roots);
  return out;
}
} // namespace at::native
