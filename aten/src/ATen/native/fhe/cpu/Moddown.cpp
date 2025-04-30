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
#include "ATen/native/fhe/cpu/KeySwitch.h"
#include "ATen/native/fhe/cpu/NttImpl.h"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
namespace fhe {
void negateInplace_(
    size_t degree,
    size_t log_degree,
    size_t batch,
    const uint64_t* primes,
    uint64_t* op) {
  for (int i = 0; i < batch * degree; i++) {
    const int prime_idx = i >> log_degree;
    const uint64_t prime = primes[prime_idx];
    if (op[i] != 0)
      op[i] = prime - op[i];
  }
}
void moddown_kernel(
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
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int out_prime_idx = 0; out_prime_idx < end_length; out_prime_idx++) {
    const auto prime = d_primes[out_prime_idx];
    const auto barret_ratio = d_barret_ratio[out_prime_idx];
    const auto barret_k = d_barret_k[out_prime_idx];
    for (int degree_idx = 0; degree_idx < degree_; degree_idx++) {
      __uint128_t accum = accumulate_in_modup(
          ptr, degree_, hat_mod_end, start_length, degree_idx, out_prime_idx);

      uint64_t out =
          barret_reduction_128_64(accum, prime, barret_ratio, barret_k);
      to[out_prime_idx * degree_ + degree_idx] = out;
    }
  }
}
} // namespace fhe
namespace at::native {

static void NegateInplace(
    uint64_t* op1,
    const int batch,
    const Tensor& primes,
    const int64_t param_degree,
    const int64_t param_log_degree) {
  AT_DISPATCH_V2(
      kUInt64,
      "NegateInplace",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * batch / block_dim;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        fhe::negateInplace_(
            param_degree, param_log_degree, batch, primes_ptr, op1);
      }),
      kUInt64);
}
static void moddown_impl(
    uint64_t* from_ptr,
    const int64_t param_degree,
    const int64_t param_log_degree,
    const int64_t param_alpha_,
    const int64_t start_length,
    const int64_t end_length,
    const Tensor& primes,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    uint64_t* to_ptr) {
  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown[0];
  AT_DISPATCH_V2(
      kUInt64,
      "moddownImpl",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * end_length / block_dim;
        auto ptr = from_ptr + param_degree * end_length;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto param_barret_ratio_ptr = reinterpret_cast<uint64_t*>(
            param_barret_ratio.data_ptr<uint64_t>());
        auto param_barret_k_ptr =
            reinterpret_cast<uint64_t*>(param_barret_k.data_ptr<uint64_t>());
        auto prod_q_i_mod_q_j_ptr =
            reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
        fhe::moddown_kernel(
            param_degree,
            primes_ptr,
            param_barret_ratio_ptr,
            param_barret_k_ptr,
            param_log_degree,
            ptr,
            prod_q_i_mod_q_j_ptr,
            start_length * end_length,
            param_alpha_,
            end_length,
            to_ptr);
      }),
      kUInt64);
}

static void moddown_cpu_template(
    // aten::moddown
    const Tensor& from,
    Tensor& workspace,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    Tensor& res) {
  const int start_length = alpha;
  const int end_length = curr_limbs;

  auto hat_inverse_vec = hat_inverse_vec_moddown[0];
  auto hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown[0];
  auto workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace.data_ptr<uint64_t>());
  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  iNTT_impl(
      workspace_ptr,
      workspace_ptr,
      end_length,
      start_length,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch_(
      workspace_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      level,
      alpha,
      curr_limbs,
      0,
      param_degree,
      workspace_ptr,
      param_primes);

  moddown_impl(
      workspace_ptr,
      param_degree,
      param_log_degree,
      alpha,
      start_length,
      end_length,
      param_primes,
      prod_q_i_mod_q_j_moddown,
      param_barret_ratio,
      param_barret_k,
      to_ptr);

  NTT_impl(
      to_ptr,
      to_ptr,
      0,
      end_length,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  const auto& prod_inv = prod_inv_moddown[0];
  const auto& prod_inv_psinv = prod_inv_shoup_moddown[0];

  SubInplace(to_ptr, from_ptr, end_length, param_degree, param_primes);

  NegateInplace(
      to_ptr, end_length, param_primes, param_degree, param_log_degree);

  const_mult_batch_(
      to_ptr,
      prod_inv,
      prod_inv_psinv,
      0,
      end_length,
      0,
      0,
      param_degree,
      to_ptr,
      param_primes);
}

Tensor moddown_cpu(
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
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  auto res = at::empty({curr_limbs * N}, in.options());
  auto workspace = in.clone();
  moddown_cpu_template(
      workspace,
      workspace,
      curr_limbs,
      L,
      sizeP,
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
      res);
  return res;
}
} // namespace at::native