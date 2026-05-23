#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <omp.h>

#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/NttImpl.h"
#include "ATen/native/fhe/cpu/arithmetic.h"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace {

using at::Tensor;
using at::native::NTT_impl;
using at::native::const_mult_batch;
using at::native::iNTT_impl;
using at::native::switch_modulus;

void const_mult_add_batch(
    uint64_t* out_ptr,
    const uint64_t* op1_ptr,
    const uint64_t* cnst_ptr,
    const uint64_t* cnst_psinv_ptr,
    int64_t batch,
    int64_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* primes_ptr) {
  const size_t L_OUTN = L_OUT * static_cast<size_t>(N);
  const size_t BL_OUTN = L_OUTN * num_cipher;
  const size_t L_INN = L_IN * static_cast<size_t>(N);
  const size_t BL_INN = L_INN * num_cipher;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (size_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (size_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      for (int64_t limb = 0; limb < batch; ++limb) {
        uint64_t* to = out_ptr + cv_id * BL_OUTN + cipher_id * L_OUTN + limb * N;
        const uint64_t* from = op1_ptr + cv_id * BL_INN + cipher_id * L_INN + limb * N;

        const uint64_t prime = primes_ptr[limb];
        const uint64_t cnst = cnst_ptr[limb];
        const uint64_t cnst_psinv = cnst_psinv_ptr[limb];

        for (int64_t n = 0; n < N; ++n) {
          uint64_t out = fhe::mul_and_reduce_shoup(from[n], cnst, cnst_psinv, prime);
          if (out >= prime) {
            out -= prime;
          }
          to[n] = fhe::add_mod(to[n], out, prime);
        }
      }
    }
  }
}

void rescale_one_level_template(
    Tensor& res,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    uint64_t old_primes,
    const Tensor& param_primes,
    const Tensor& switch_modulus_map,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    Tensor& workspace) {
  const int64_t end_length = curr_limbs - 1;

  const auto num_cv = from.sizes()[0];
  const auto num_cipher = from.sizes()[1];
  const auto L_IN = from.sizes()[2];
  const auto L_OUT = res.sizes()[2];

  auto* from_ptr = from.data_ptr<uint64_t>();
  auto* workspace_ptr = workspace.data_ptr<uint64_t>();
  auto* to_ptr = res.data_ptr<uint64_t>();

  iNTT_impl(
      workspace_ptr + N * end_length,
      from_ptr + N * end_length,
      1,
      N,
      L_IN,
      L_IN,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>() + end_length,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + end_length * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() + end_length * N);

  switch_modulus(
      to_ptr,
      workspace_ptr + N * end_length,
      curr_limbs - 1,
      curr_limbs - 1,
      N,
      L_OUT,
      L_IN,
      num_cv,
      num_cipher,
      old_primes >> 1,
      param_primes,
      switch_modulus_map);

  const int64_t start_op2_idx = (L - curr_limbs) * (L - 1);
  const_mult_batch(
      to_ptr,
      to_ptr,
      qlql_inv_mod_ql_div_ql_mod_q.data_ptr<uint64_t>() + start_op2_idx,
      qlql_inv_mod_ql_div_ql_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
      curr_limbs - 1,
      N,
      L_OUT,
      L_OUT,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>());

  NTT_impl(
      to_ptr,
      end_length,
      N,
      L_OUT,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());

  const int64_t start_q_inv = (curr_limbs - 1) * L;
  const_mult_add_batch(
      to_ptr,
      from_ptr,
      q_inv_mod_q.data_ptr<uint64_t>() + start_q_inv,
      q_inv_mod_q_shoup.data_ptr<uint64_t>() + start_q_inv,
      end_length,
      N,
      L_OUT,
      L_IN,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>());
}

} // namespace

namespace at::native {

Tensor rescale_one_level_cpu(
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
    const Tensor& q_inv_mod_q_shoup) {
  TORCH_INTERNAL_ASSERT(from.dim() == 4);
  const auto num_cv = from.sizes()[0];
  const auto batch = from.sizes()[1];

  auto res = at::empty({num_cv, batch, curr_limbs - 1, N}, from.options());
  auto workspace = at::empty({num_cv, batch, curr_limbs, N}, from.options());

  rescale_one_level_template(
      res,
      from,
      curr_limbs,
      l,
      L,
      N,
      static_cast<uint64_t>(old_primes),
      param_primes,
      switch_modulus_map,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      workspace);

  return res;
}

} // namespace at::native
