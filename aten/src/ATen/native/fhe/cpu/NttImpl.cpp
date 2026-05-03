#include "ATen/native/fhe/cpu/NttImpl.h"

#include <omp.h>
#ifdef USE_AVX512
#include <immintrin.h>
#endif

namespace {

inline int get_msb(size_t x) {
  if (x == 0) {
    return -1;
  }
  int position = 0;
  while (x > 0) {
    x >>= 1;
    ++position;
  }
  return position;
}

inline void intt_single_poly(
    uint64_t* out,
    const uint64_t* in,
    size_t N,
    uint64_t modulus,
    const uint64_t* inverse_power_of_roots_div_two,
    const uint64_t* inverse_scaled_power_of_roots_div_two) {
  if (N <= 1) {
    if (N == 1) {
      out[0] = in[0];
    }
    return;
  }

  const uint32_t n = static_cast<uint32_t>(N);

  {
    const uint32_t m0 = n >> 1;
    const uint32_t t0 = 1;
    const uint32_t logt0 = 1;
    for (uint32_t i = 0; i < m0; ++i) {
      const auto omega = inverse_power_of_roots_div_two[i + m0];
      const auto preconOmega = inverse_scaled_power_of_roots_div_two[i + m0];
      const uint32_t j = (i << logt0);
      auto loVal = in[j];
      auto hiVal = in[j + t0];
      fhe::butt_intt_local(loVal, hiVal, omega, preconOmega, modulus);
      out[j] = loVal;
      out[j + t0] = hiVal;
    }
  }

  for (uint32_t m = n >> 2, t = 2, logt = 2; m > 1; m >>= 1, t <<= 1, ++logt) {
    for (uint32_t i = 0; i < m; ++i) {
      const auto omega = inverse_power_of_roots_div_two[i + m];
      const auto preconOmega = inverse_scaled_power_of_roots_div_two[i + m];
      uint32_t j1 = (i << logt);
      const uint32_t j2 = j1 + t;
#ifdef USE_AVX512
      const __m512i vec_omega = _mm512_set1_epi64(omega);
      const __m512i vec_precon = _mm512_set1_epi64(preconOmega);
      const __m512i vec_modulus = _mm512_set1_epi64(modulus);
      for (; j1 + 8 <= j2; j1 += 8) {
        __m512i vec_lo = _mm512_loadu_si512(&out[j1]);
        __m512i vec_hi = _mm512_loadu_si512(&out[j1 + t]);
        fhe::butt_intt_local_avx512(vec_lo, vec_hi, vec_omega, vec_precon, vec_modulus);
        _mm512_storeu_si512(&out[j1], vec_lo);
        _mm512_storeu_si512(&out[j1 + t], vec_hi);
      }
#endif
      for (; j1 < j2; ++j1) {
        auto loVal = out[j1];
        auto hiVal = out[j1 + t];
        fhe::butt_intt_local(loVal, hiVal, omega, preconOmega, modulus);
        out[j1] = loVal;
        out[j1 + t] = hiVal;
      }
    }
  }

  const auto omega = inverse_power_of_roots_div_two[1];
  const auto preconOmega = inverse_scaled_power_of_roots_div_two[1];
  const uint32_t j2 = n >> 1;
  uint32_t j1 = 0;
#ifdef USE_AVX512
  const __m512i vec_omega = _mm512_set1_epi64(omega);
  const __m512i vec_precon = _mm512_set1_epi64(preconOmega);
  const __m512i vec_modulus = _mm512_set1_epi64(modulus);
  for (; j1 + 8 <= j2; j1 += 8) {
    __m512i vec_lo = _mm512_loadu_si512(&out[j1]);
    __m512i vec_hi = _mm512_loadu_si512(&out[j1 + j2]);
    fhe::butt_intt_local_avx512(vec_lo, vec_hi, vec_omega, vec_precon, vec_modulus);
    const __mmask8 lo_gt = _mm512_cmp_epu64_mask(vec_lo, vec_modulus, _MM_CMPINT_GT);
    const __mmask8 hi_gt = _mm512_cmp_epu64_mask(vec_hi, vec_modulus, _MM_CMPINT_GT);
    vec_lo = _mm512_mask_sub_epi64(vec_lo, lo_gt, vec_lo, vec_modulus);
    vec_hi = _mm512_mask_sub_epi64(vec_hi, hi_gt, vec_hi, vec_modulus);
    _mm512_storeu_si512(&out[j1], vec_lo);
    _mm512_storeu_si512(&out[j1 + j2], vec_hi);
  }
#endif
  for (; j1 < j2; ++j1) {
    auto loVal = out[j1];
    auto hiVal = out[j1 + j2];
    fhe::butt_intt_local(loVal, hiVal, omega, preconOmega, modulus);

    for (int k = 0; k < 8; ++k) {
      if (loVal > modulus) {
        loVal -= modulus;
      }
      if (hiVal > modulus) {
        hiVal -= modulus;
      }
    }

    out[j1] = loVal;
    out[j1 + j2] = hiVal;
  }
}

inline void ntt_single_poly(
    uint64_t* data,
    size_t N,
    uint64_t modulus,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup) {
  if (N <= 1) {
    return;
  }

  const int64_t n = static_cast<int64_t>(N) >> 1;
  for (uint32_t m = 1, t = static_cast<uint32_t>(n), logt = static_cast<uint32_t>(get_msb(t));
       m < n;
       m <<= 1, t >>= 1, --logt) {
    for (uint32_t i = 0; i < m; ++i) {
      const auto omega = power_of_roots[i + m];
      const auto preconOmega = power_of_roots_shoup[i + m];
      uint32_t j1 = (i << logt);
      const uint32_t j2 = j1 + t;
#ifdef USE_AVX512
      const __m512i vec_omega = _mm512_set1_epi64(omega);
      const __m512i vec_precon = _mm512_set1_epi64(preconOmega);
      const __m512i vec_modulus = _mm512_set1_epi64(modulus);
      for (; j1 + 8 <= j2; j1 += 8) {
        __m512i vec_a = _mm512_loadu_si512(&data[j1]);
        __m512i vec_b = _mm512_loadu_si512(&data[j1 + t]);
        fhe::butt_ntt_local_avx512(vec_a, vec_b, vec_omega, vec_precon, vec_modulus);
        _mm512_storeu_si512(&data[j1], vec_a);
        _mm512_storeu_si512(&data[j1 + t], vec_b);
      }
#endif
      for (; j1 < j2; ++j1) {
        auto a1 = data[j1];
        auto b1 = data[j1 + t];
        fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
        data[j1] = a1;
        data[j1 + t] = b1;
      }
    }
  }

  for (uint32_t i = 0; i < (n << 1); i += 2) {
    const auto omega = power_of_roots[(i >> 1) + n];
    const auto preconOmega = power_of_roots_shoup[(i >> 1) + n];
    auto a1 = data[i];
    auto b1 = data[i + 1];
    fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);

    for (int k = 0; k < 3; ++k) {
      if (a1 > modulus) {
        a1 -= modulus;
      }
      if (b1 > modulus) {
        b1 -= modulus;
      }
    }

    data[i] = a1;
    data[i + 1] = b1;
  }
}

} // namespace

namespace at::native {

void iNTT_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t num_batch,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* inverse_power_of_roots_div_two,
    const uint64_t* inverse_scaled_power_of_roots_div_two) {
  TORCH_INTERNAL_ASSERT(num_cv == 1 || num_cv == 2, "Unsupported num_cv");

  const size_t L_OUTN = L_OUT * N;
  const size_t BL_OUTN = L_OUTN * num_cipher;
  const size_t L_INN = L_IN * N;
  const size_t BL_INN = L_INN * num_cipher;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (size_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (size_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      for (size_t batch_id = 0; batch_id < num_batch; ++batch_id) {
        uint64_t* out_base = out_ptr + cv_id * BL_OUTN + cipher_id * L_OUTN + batch_id * N;
        uint64_t* in_base = in_ptr + cv_id * BL_INN + cipher_id * L_INN + batch_id * N;

        intt_single_poly(
            out_base,
            in_base,
            N,
            param_primes[batch_id],
            inverse_power_of_roots_div_two + batch_id * N,
            inverse_scaled_power_of_roots_div_two + batch_id * N);
      }
    }
  }
}

void NTT_impl(
    uint64_t* inout_ptr,
    size_t num_batch,
    size_t N,
    size_t L,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes,
    const uint64_t* param_power_of_roots_shoup,
    const uint64_t* param_power_of_roots) {
  TORCH_INTERNAL_ASSERT(num_cv == 1 || num_cv == 2, "Unsupported num_cv");

  const size_t LN = L * N;
  const size_t BLN = LN * num_cipher;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (size_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (size_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      for (size_t batch_id = 0; batch_id < num_batch; ++batch_id) {
        uint64_t* data = inout_ptr + cv_id * BLN + cipher_id * LN + batch_id * N;
        ntt_single_poly(
            data,
            N,
            param_primes[batch_id],
            param_power_of_roots + batch_id * N,
            param_power_of_roots_shoup + batch_id * N);
      }
    }
  }
}

void iNTT_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  const auto* inv_ptr = inverse_power_of_roots_div_two.data_ptr<uint64_t>();
  const auto* inv_scaled_ptr = inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>();
  const auto* primes_ptr = param_primes.data_ptr<uint64_t>();
  const int64_t gap = level - curr_limbs;

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t primeidx = start_prime_idx + b;
    const int64_t prime_idx =
        primeidx + (((primeidx >= 0) && (primeidx < curr_limbs)) ? 0 : gap);
    iNTT_impl(
        out_ptr + b * param_degree,
        in_ptr + b * param_degree,
        1,
        static_cast<size_t>(param_degree),
        1,
        1,
        1,
        1,
        primes_ptr + prime_idx,
        inv_ptr + prime_idx * param_degree,
        inv_scaled_ptr + prime_idx * param_degree);
  }
}

void NTT_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  const auto* roots_shoup_ptr = param_power_of_roots_shoup.data_ptr<uint64_t>();
  const auto* roots_ptr = param_power_of_roots.data_ptr<uint64_t>();
  const auto* primes_ptr = param_primes.data_ptr<uint64_t>();

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t prime_idx = start_prime_idx + b;
    NTT_impl(
        op_ptr + b * param_degree,
        1,
        static_cast<size_t>(param_degree),
        1,
        1,
        1,
        primes_ptr + prime_idx,
        roots_shoup_ptr + prime_idx * param_degree,
        roots_ptr + prime_idx * param_degree);
  }
}

void NTT_except_some_range_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    int64_t excluded_range_start,
    int64_t excluded_range_size,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  const auto* roots_shoup_ptr = param_power_of_roots_shoup.data_ptr<uint64_t>();
  const auto* roots_ptr = param_power_of_roots.data_ptr<uint64_t>();
  const auto* primes_ptr = param_primes.data_ptr<uint64_t>();
  const int64_t gap = level - curr_limbs;
  const int64_t excluded_range_end = excluded_range_start + excluded_range_size;

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t primeidx = start_prime_idx + b;
    if (primeidx >= excluded_range_start && primeidx < excluded_range_end) {
      continue;
    }
    const int64_t prime_idx =
        primeidx + (((primeidx >= 0) && (primeidx < curr_limbs)) ? 0 : gap);

    NTT_impl(
        op_ptr + primeidx * param_degree,
        1,
        static_cast<size_t>(param_degree),
        1,
        1,
        1,
        primes_ptr + prime_idx,
        roots_shoup_ptr + prime_idx * param_degree,
        roots_ptr + prime_idx * param_degree);
  }
}

} // namespace at::native
