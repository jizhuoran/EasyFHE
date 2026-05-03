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

namespace fhe {

__uint128_t accumulate_in_modup(
    const uint64_t* ptr,
    const int degree,
    const uint64_t* hat_mod_end,
    const int start_length,
    const int degree_idx,
    const int hat_mod_end_idx) {
  __uint128_t accum{0};
  for (int i = 0; i < start_length; ++i) {
    const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
    accum += static_cast<__uint128_t>(ptr[i * degree + degree_idx]) * op2;
  }
  return accum;
}

} // namespace fhe

namespace at::native {

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
    const uint64_t* primes_ptr) {
  TORCH_INTERNAL_ASSERT(num_cv == 1 || num_cv == 2, "Unsupported num_cv");

  const size_t L_OUTN = L_OUT * N;
  const size_t BL_OUTN = L_OUTN * num_cipher;
  const size_t L_INN = L_IN * N;
  const size_t BL_INN = L_INN * num_cipher;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (size_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (size_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      for (size_t limb = 0; limb < batch; ++limb) {
        uint64_t* out_base =
            res_ptr + cv_id * BL_OUTN + cipher_id * L_OUTN + limb * N;
        const uint64_t* in_base =
            op1_ptr + cv_id * BL_INN + cipher_id * L_INN + limb * N;

        const uint64_t prime = primes_ptr[limb];
        const uint64_t cnst = op2_ptr[limb];
        const uint64_t cnst_psinv = op2_psinv_ptr[limb];

#ifdef USE_AVX512
        const __m512i prime_vec = _mm512_set1_epi64(prime);
        const __m512i cnst_vec = _mm512_set1_epi64(cnst);
        const __m512i cnst_psinv_vec = _mm512_set1_epi64(cnst_psinv);
        size_t n = 0;
        for (; n + 8 <= N; n += 8) {
          const __m512i in_vec = _mm512_loadu_si512(&in_base[n]);
          __m512i out_vec = fhe::mul_and_reduce_shoup_avx512_full(
              in_vec, cnst_vec, cnst_psinv_vec, prime_vec);
          const __mmask8 ge_mask =
              _mm512_cmp_epu64_mask(out_vec, prime_vec, _MM_CMPINT_GE);
          out_vec = _mm512_mask_sub_epi64(out_vec, ge_mask, out_vec, prime_vec);
          _mm512_storeu_si512(&out_base[n], out_vec);
        }
        for (; n < N; ++n) {
          uint64_t out = fhe::mul_and_reduce_shoup(in_base[n], cnst, cnst_psinv, prime);
          if (out >= prime) {
            out -= prime;
          }
          out_base[n] = out;
        }
#else
        for (size_t n = 0; n < N; ++n) {
          uint64_t out = fhe::mul_and_reduce_shoup(in_base[n], cnst, cnst_psinv, prime);
          if (out >= prime) {
            out -= prime;
          }
          out_base[n] = out;
        }
#endif
      }
    }
  }
}

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
    const Tensor& switch_modulus_map) {
  TORCH_INTERNAL_ASSERT(num_cv == 1 || num_cv == 2, "Unsupported num_cv");

  const auto* primes_ptr = primes.data_ptr<uint64_t>();
  const auto* switch_modulus_map_ptr = switch_modulus_map.data_ptr<uint64_t>();
  const auto* diffs = switch_modulus_map_ptr + old_prime_index * primes.numel();

  const size_t L_OUTN = L_OUT * static_cast<size_t>(N);
  const size_t BL_OUTN = L_OUTN * num_cipher;
  const size_t L_INN = L_IN * static_cast<size_t>(N);
  const size_t BL_INN = L_INN * num_cipher;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (size_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (size_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      for (int64_t limb = 0; limb < batch; ++limb) {
        uint64_t* out_base =
            out_ptr + cv_id * BL_OUTN + cipher_id * L_OUTN + limb * N;
        const uint64_t* in_base = in_ptr + cv_id * BL_INN + cipher_id * L_INN;

        const uint64_t new_modulus = primes_ptr[limb];
        const uint64_t diff = diffs[limb];
        for (int64_t n = 0; n < N; ++n) {
          const uint64_t in_val = in_base[n];
          uint64_t res = in_val + (in_val > old_modulus_by_two ? diff : 0);
          if (res >= new_modulus) {
            res -= new_modulus;
          }
          out_base[n] = res;
        }
      }
    }
  }
}

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
    const Tensor& primes) {
  const auto* op2_ptr = op2.data_ptr<uint64_t>();
  const auto* op2_psinv_ptr = op2_psinv.data_ptr<uint64_t>();
  const auto* primes_ptr = primes.data_ptr<uint64_t>();

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int64_t limb = 0; limb < batch; ++limb) {
    const int64_t op2_idx = start_op2_idx + limb;
    const int64_t prime_idx = start_prime_idx + limb;
    const int64_t base_idx = (start_op1_idx + limb) * param_degree;
    const uint64_t prime = primes_ptr[prime_idx];
    const uint64_t cnst = op2_ptr[op2_idx];
    const uint64_t cnst_psinv = op2_psinv_ptr[op2_idx];

#ifdef USE_AVX512
    const __m512i prime_vec = _mm512_set1_epi64(prime);
    const __m512i cnst_vec = _mm512_set1_epi64(cnst);
    const __m512i cnst_psinv_vec = _mm512_set1_epi64(cnst_psinv);
    int64_t n = 0;
    for (; n + 8 <= param_degree; n += 8) {
      const __m512i in_vec = _mm512_loadu_si512(&op1_ptr[base_idx + n]);
      __m512i out_vec =
          fhe::mul_and_reduce_shoup_avx512_full(in_vec, cnst_vec, cnst_psinv_vec, prime_vec);
      const __mmask8 ge_mask =
          _mm512_cmp_epu64_mask(out_vec, prime_vec, _MM_CMPINT_GE);
      out_vec = _mm512_mask_sub_epi64(out_vec, ge_mask, out_vec, prime_vec);
      _mm512_storeu_si512(&res_ptr[base_idx + n], out_vec);
    }
    for (; n < param_degree; ++n) {
      uint64_t out = fhe::mul_and_reduce_shoup(op1_ptr[base_idx + n], cnst, cnst_psinv, prime);
      if (out >= prime) {
        out -= prime;
      }
      res_ptr[base_idx + n] = out;
    }
#else
    for (int64_t n = 0; n < param_degree; ++n) {
      uint64_t out = fhe::mul_and_reduce_shoup(op1_ptr[base_idx + n], cnst, cnst_psinv, prime);
      if (out >= prime) {
        out -= prime;
      }
      res_ptr[base_idx + n] = out;
    }
#endif
  }
}

void vec_mod_batch(
    uint64_t* op1_ptr,
    const Tensor& primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    int64_t batch,
    int64_t degree,
    uint64_t* res_ptr) {
  const auto* primes_ptr = primes.data_ptr<uint64_t>();
  const auto* barret_ratio_ptr = param_barret_ratio.data_ptr<uint64_t>();
  const auto* barret_k_ptr = param_barret_k.data_ptr<uint64_t>();

  const int64_t total = batch * degree;
  const int max_threads = omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int64_t i = 0; i < total; ++i) {
    const int64_t out_prime_idx = i / degree;
    const int64_t op1_idx = i % degree;
    const auto prime = primes_ptr[out_prime_idx];
    const auto barret_ratio = barret_ratio_ptr[out_prime_idx];
    const auto barret_k = barret_k_ptr[out_prime_idx];
    res_ptr[i] = fhe::barret_reduction_64_64(
        op1_ptr[op1_idx], prime, barret_ratio, static_cast<unsigned>(barret_k));
  }
}

void switch_modulus(
    uint64_t* ptr,
    uint64_t* res_ptr,
    const Tensor& primes,
    int64_t old_prime_index,
    int64_t batch,
    int64_t degree) {
  const auto* primes_ptr = primes.data_ptr<uint64_t>();
  const uint64_t old_modulus = primes_ptr[old_prime_index];
  const uint64_t old_modulus_by_two = old_modulus >> 1;

  const int64_t total = batch * degree;
  const int max_threads = omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int64_t i = 0; i < total; ++i) {
    const int64_t new_modulus_idx = i / degree;
    const uint64_t new_modulus = primes_ptr[new_modulus_idx];

    const uint64_t modulus_diff =
        (old_modulus > new_modulus)
        ? (new_modulus - (old_modulus % new_modulus))
        : (new_modulus - old_modulus);

    const int64_t input_idx = i % degree;
    uint64_t val = ptr[input_idx] + (ptr[input_idx] > old_modulus_by_two ? modulus_diff : 0);
    if (new_modulus <= old_modulus) {
      val %= new_modulus;
    }
    res_ptr[i] = val;
  }
}

} // namespace at::native
