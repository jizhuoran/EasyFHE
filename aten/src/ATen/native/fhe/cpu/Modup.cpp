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

#include <cstring>

#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/NttImpl.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace {

using at::Tensor;
using at::native::NTT_impl;
using at::native::const_mult_batch;
using at::native::iNTT_impl;

void modup_step_two_cpu(
    uint64_t* to,
    const uint64_t* from,
    int64_t begin_idx,
    int64_t N,
    int64_t alpha,
    int64_t curr_limbs,
    int64_t L,
    int64_t group_size,
    const uint64_t* primes,
    const uint64_t* barret_ratios,
    const uint64_t* barret_ks,
    const uint64_t* hat_mod_end,
    int64_t end_length) {
  const int64_t gap = L - curr_limbs;
  const int max_threads = omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int64_t out_iter = 0; out_iter < end_length; ++out_iter) {
    const int64_t out_idx = out_iter + ((out_iter >= begin_idx) ? group_size : 0);
    const int64_t prime_idx =
        out_idx +
        (((out_idx >= 0 && out_idx < begin_idx) ||
          (out_idx >= (begin_idx + group_size) && out_idx < curr_limbs))
             ? 0
             : gap);
    const auto prime = primes[prime_idx];
    const auto barret_ratio = barret_ratios[prime_idx];
    const auto barret_k = barret_ks[prime_idx];

#ifdef USE_AVX512
    const __m512i prime_vec = _mm512_set1_epi64(prime);
    const __m512i ratio_vec = _mm512_set1_epi64(barret_ratio);
    int64_t degree_idx = 0;
    for (; degree_idx + 8 <= N; degree_idx += 8) {
      __m512i accum_lo = _mm512_setzero_si512();
      __m512i accum_hi = _mm512_setzero_si512();
      for (int64_t i = 0; i < group_size; ++i) {
        const __m512i op1_vec = _mm512_loadu_si512(&from[i * N + degree_idx]);
        const __m512i op2_vec = _mm512_set1_epi64(hat_mod_end[out_iter * alpha + i]);
        const __m512i mul_lo = _mm512_mullo_epi64(op1_vec, op2_vec);
        const __m512i mul_hi = fhe::avx_umul64hi(op1_vec, op2_vec);
        fhe::avx512_add_u128(accum_lo, accum_hi, mul_lo, mul_hi, accum_lo, accum_hi);
      }
      const __m512i out_vec = fhe::barret_reduction_128_64_avx512(
          accum_lo, accum_hi, prime_vec, ratio_vec, static_cast<unsigned>(barret_k));
      _mm512_storeu_si512(&to[out_idx * N + degree_idx], out_vec);
    }
    for (; degree_idx < N; ++degree_idx) {
      __uint128_t accum{0};
      for (int64_t i = 0; i < group_size; ++i) {
        const uint64_t op1 = from[i * N + degree_idx];
        const uint64_t op2 = hat_mod_end[out_iter * alpha + i];
        accum += static_cast<__uint128_t>(op1) * op2;
      }
      to[out_idx * N + degree_idx] = fhe::barret_reduction_128_64(
          accum, prime, barret_ratio, static_cast<unsigned>(barret_k));
    }
#else
    for (int64_t degree_idx = 0; degree_idx < N; ++degree_idx) {
      __uint128_t accum{0};
      for (int64_t i = 0; i < group_size; ++i) {
        const uint64_t op1 = from[i * N + degree_idx];
        const uint64_t op2 = hat_mod_end[out_iter * alpha + i];
        accum += static_cast<__uint128_t>(op1) * op2;
      }
      to[out_idx * N + degree_idx] = fhe::barret_reduction_128_64(
          accum, prime, barret_ratio, static_cast<unsigned>(barret_k));
    }
#endif
  }
}

void modup_matmul(
    uint64_t* to_ptr,
    uint64_t* from_ptr,
    int64_t beta_idx,
    int64_t alpha,
    int64_t N,
    int64_t curr_limbs,
    int64_t L,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  const int64_t sizeQP = primes.numel();
  const int64_t sizeP = sizeQP - L;
  const int64_t begin_idx = beta_idx * alpha;
  const int64_t group_size =
      ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;
  const int64_t end_length = curr_limbs + sizeP - group_size;

  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_js[beta_idx];

  modup_step_two_cpu(
      to_ptr,
      from_ptr,
      begin_idx,
      N,
      alpha,
      curr_limbs,
      L,
      group_size,
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      prod_q_i_mod_q_j.data_ptr<uint64_t>(),
      end_length);
}

void modup_cpu_template(
    Tensor& to,
    const Tensor& from,
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
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup) {
  const auto num_cipher = from.sizes()[1];
  const int64_t sizeQP = primes.numel();
  const int64_t sizeP = sizeQP - L;
  const int64_t num_moduli_after_modup = curr_limbs + sizeP;

  const auto L_OUT = to.sizes()[2];
  const auto L_IN = from.sizes()[2];

  auto* to_ptr__ = to.data_ptr<uint64_t>();
  auto* from_ptr__ = from.data_ptr<uint64_t>();

  for (int64_t group_idx = 0; group_idx < beta; ++group_idx) {
    auto* to_ptr_ = to_ptr__ + num_moduli_after_modup * N * group_idx;
    auto* from_ptr_ = from_ptr__ + alpha * N * group_idx;

    const int64_t begin_idx = group_idx * alpha;
    const int64_t in_C_L_len =
        ((begin_idx + alpha) > curr_limbs) ? (curr_limbs - begin_idx) : alpha;

    auto hat_inverse_vec = hat_inverse_vecs[group_idx * alpha + (in_C_L_len - 1)];
    auto hat_inverse_vec_psinv =
        hat_inverse_vec_shoups[group_idx * alpha + (in_C_L_len - 1)];

    iNTT_impl(
        to_ptr_ + begin_idx * N,
        from_ptr_,
        in_C_L_len,
        N,
        L_OUT,
        L_IN,
        1,
        num_cipher,
        primes.data_ptr<uint64_t>() + begin_idx,
        inverse_power_of_roots_div_two.data_ptr<uint64_t>() + begin_idx * N,
        inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() + begin_idx * N);

    const_mult_batch(
        to_ptr_ + begin_idx * N,
        to_ptr_ + begin_idx * N,
        hat_inverse_vec.data_ptr<uint64_t>(),
        hat_inverse_vec_psinv.data_ptr<uint64_t>(),
        in_C_L_len,
        N,
        L_OUT,
        L_OUT,
        1,
        num_cipher,
        primes.data_ptr<uint64_t>() + begin_idx);

    for (int64_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      auto* to_ptr = to_ptr_ + (L_OUT * N * cipher_id);
      auto* from_ptr = from_ptr_ + (L_IN * N * cipher_id);
      modup_matmul(
          to_ptr,
          to_ptr + N * begin_idx,
          group_idx,
          alpha,
          N,
          curr_limbs,
          L,
          prod_q_i_mod_q_js,
          primes,
          barret_ratio,
          barret_k);

      std::memcpy(
          to_ptr + begin_idx * N,
          from_ptr,
          static_cast<size_t>(in_C_L_len * N) * sizeof(uint64_t));
    }

    if (begin_idx > 0) {
      NTT_impl(
          to_ptr_,
          begin_idx,
          N,
          L_OUT,
          1,
          num_cipher,
          primes.data_ptr<uint64_t>(),
          power_of_roots_shoup.data_ptr<uint64_t>(),
          power_of_roots.data_ptr<uint64_t>());
    }

    if (curr_limbs - begin_idx - in_C_L_len > 0) {
      NTT_impl(
          to_ptr_ + (begin_idx + in_C_L_len) * N,
          curr_limbs - begin_idx - in_C_L_len,
          N,
          L_OUT,
          1,
          num_cipher,
          primes.data_ptr<uint64_t>() + begin_idx + in_C_L_len,
          power_of_roots_shoup.data_ptr<uint64_t>() + (begin_idx + in_C_L_len) * N,
          power_of_roots.data_ptr<uint64_t>() + (begin_idx + in_C_L_len) * N);
    }

    if (sizeP > 0) {
      NTT_impl(
          to_ptr_ + curr_limbs * N,
          sizeP,
          N,
          L_OUT,
          1,
          num_cipher,
          primes.data_ptr<uint64_t>() + L,
          power_of_roots_shoup.data_ptr<uint64_t>() + L * N,
          power_of_roots.data_ptr<uint64_t>() + L * N);
    }
  }
}

} // namespace

namespace at::native {

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
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  const auto num_cv = in.sizes()[0];
  TORCH_INTERNAL_ASSERT(num_cv == 1, "modup_cpu expects num_cv == 1");
  const auto batch = in.sizes()[1];

  const int64_t sizeQP = primes.numel();
  const int64_t sizeP = sizeQP - L;

  auto out = at::empty({num_cv, batch, beta * (curr_limbs + sizeP), N}, in.options());

  modup_cpu_template(
      out,
      in,
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
      power_of_roots,
      power_of_roots_shoup);

  return out;
}

} // namespace at::native
