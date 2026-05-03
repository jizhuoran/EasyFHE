#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <omp.h>
#ifdef USE_AVX512
#include <immintrin.h>
#endif

#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/NttImpl.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace {

using at::Tensor;
using at::native::NTT_impl;
using at::native::const_mult_batch;
using at::native::iNTT_impl;

void const_sub_mult_batch(
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
          const uint64_t val = fhe::sub_mod(from[n], to[n], prime);
          uint64_t out = fhe::mul_and_reduce_shoup(val, cnst, cnst_psinv, prime);
          if (out >= prime) {
            out -= prime;
          }
          to[n] = out;
        }
      }
    }
  }
}

void moddown_impl(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    size_t sizeP,
    size_t end_length,
    const Tensor& primes,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  const size_t L_OUTN = L_OUT * N;
  const size_t BL_OUTN = L_OUTN * num_cipher;
  const size_t L_INN = L_IN * N;
  const size_t BL_INN = L_INN * num_cipher;

  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown;
  const auto* primes_ptr = primes.data_ptr<uint64_t>();
  const auto* barret_ratio_ptr = barret_ratio.data_ptr<uint64_t>();
  const auto* barret_k_ptr = barret_k.data_ptr<uint64_t>();
  const auto* hat_mod_end = prod_q_i_mod_q_j.data_ptr<uint64_t>();

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (size_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (size_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      for (size_t out_idx = 0; out_idx < end_length; ++out_idx) {
        uint64_t* to = to_ptr + cv_id * BL_OUTN + cipher_id * L_OUTN + out_idx * N;
        const uint64_t* ptr =
            from_ptr + cv_id * BL_INN + cipher_id * L_INN + end_length * N;

        const auto prime = primes_ptr[out_idx];
        const auto ratio = barret_ratio_ptr[out_idx];
        const auto k = barret_k_ptr[out_idx];

#ifdef USE_AVX512
        const __m512i prime_vec = _mm512_set1_epi64(prime);
        const __m512i ratio_vec = _mm512_set1_epi64(ratio);
        size_t degree_idx = 0;
        for (; degree_idx + 8 <= N; degree_idx += 8) {
          __m512i accum_lo = _mm512_setzero_si512();
          __m512i accum_hi = _mm512_setzero_si512();
          for (size_t i = 0; i < sizeP; ++i) {
            const __m512i op1_vec = _mm512_loadu_si512(&ptr[i * N + degree_idx]);
            const __m512i op2_vec = _mm512_set1_epi64(hat_mod_end[out_idx * sizeP + i]);
            const __m512i mul_lo = _mm512_mullo_epi64(op1_vec, op2_vec);
            const __m512i mul_hi = fhe::avx_umul64hi(op1_vec, op2_vec);
            fhe::avx512_add_u128(accum_lo, accum_hi, mul_lo, mul_hi, accum_lo, accum_hi);
          }
          const __m512i out_vec = fhe::barret_reduction_128_64_avx512(
              accum_lo, accum_hi, prime_vec, ratio_vec, static_cast<unsigned>(k));
          _mm512_storeu_si512(&to[degree_idx], out_vec);
        }
        for (; degree_idx < N; ++degree_idx) {
          __uint128_t accum{0};
          for (size_t i = 0; i < sizeP; ++i) {
            const uint64_t op1 = ptr[i * N + degree_idx];
            const uint64_t op2 = hat_mod_end[out_idx * sizeP + i];
            accum += static_cast<__uint128_t>(op1) * op2;
          }
          to[degree_idx] = fhe::barret_reduction_128_64(
              accum, prime, ratio, static_cast<unsigned>(k));
        }
#else
        for (size_t degree_idx = 0; degree_idx < N; ++degree_idx) {
          __uint128_t accum{0};
          for (size_t i = 0; i < sizeP; ++i) {
            const uint64_t op1 = ptr[i * N + degree_idx];
            const uint64_t op2 = hat_mod_end[out_idx * sizeP + i];
            accum += static_cast<__uint128_t>(op1) * op2;
          }

          to[degree_idx] =
              fhe::barret_reduction_128_64(accum, prime, ratio, static_cast<unsigned>(k));
        }
#endif
      }
    }
  }
}

void moddown_cpu_template(
    Tensor& res,
    const Tensor& from,
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
    const Tensor& inverse_scaled_power_of_roots_div_two,
    Tensor& workspace) {
  (void)log_degree;
  const int64_t start_length = sizeP;
  const int64_t end_length = curr_limbs;

  const auto num_cv = from.sizes()[0];
  const auto batch = from.sizes()[1];
  const auto L_IN = from.sizes()[2];
  const auto L_OUT = res.sizes()[2];

  auto* from_ptr = from.data_ptr<uint64_t>();
  auto* workspace_ptr = workspace.data_ptr<uint64_t>();
  auto* to_ptr = res.data_ptr<uint64_t>();

  iNTT_impl(
      workspace_ptr + curr_limbs * N,
      from_ptr + curr_limbs * N,
      start_length,
      N,
      L_IN,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>() + L,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + L * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() + L * N);

  const_mult_batch(
      workspace_ptr + curr_limbs * N,
      workspace_ptr + curr_limbs * N,
      hat_inverse_vec_moddown.data_ptr<uint64_t>(),
      hat_inverse_vec_shoup_moddown.data_ptr<uint64_t>(),
      sizeP,
      N,
      L_IN,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>() + L);

  moddown_impl(
      to_ptr,
      workspace_ptr,
      N,
      L_OUT,
      L_IN,
      num_cv,
      batch,
      sizeP,
      end_length,
      primes,
      prod_q_i_mod_q_j_moddown,
      barret_ratio,
      barret_k);

  NTT_impl(
      to_ptr,
      end_length,
      N,
      L_OUT,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  const_sub_mult_batch(
      to_ptr,
      from_ptr,
      prod_inv_moddown.data_ptr<uint64_t>(),
      prod_inv_shoup_moddown.data_ptr<uint64_t>(),
      end_length,
      N,
      L_OUT,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>());
}

} // namespace

namespace at::native {

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
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  TORCH_INTERNAL_ASSERT(in.sizes()[2] == (curr_limbs + sizeP));

  const auto num_cv = in.sizes()[0];
  const auto batch = in.sizes()[1];

  auto out = at::empty({num_cv, batch, curr_limbs, N}, in.options());
  auto workspace = at::empty({num_cv, batch, curr_limbs + sizeP, N}, in.options());

  moddown_cpu_template(
      out,
      in,
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
      workspace);

  return out;
}

} // namespace at::native
