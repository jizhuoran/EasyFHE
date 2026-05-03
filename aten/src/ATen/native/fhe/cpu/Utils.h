#pragma once

#ifdef USE_AVX512
#include <immintrin.h>
#endif
#include <cstdint>

#define STRIDED_LOOP_START(N, i) for (int i = 0; i < N; i++) {
#define STRIDED_LOOP_END }
#define TO_PTR(x) (x.data())

static inline __attribute__((always_inline)) void __syncthreads() {
  // Dummy implementation
}

namespace fhe {

using uint128_t = __uint128_t;

#ifdef USE_AVX512
static const __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffffULL);

inline void avx512_add_u128(
    __m512i lo,
    __m512i hi,
    __m512i add_lo,
    __m512i add_hi,
    __m512i& res_lo,
    __m512i& res_hi) {
  const __m512i sum_lo = _mm512_add_epi64(lo, add_lo);
  const __mmask8 carry_mask = _mm512_cmp_epu64_mask(sum_lo, lo, _MM_CMPINT_LT);
  __m512i sum_hi = _mm512_add_epi64(hi, add_hi);
  const __m512i carry_vec = _mm512_maskz_set1_epi64(carry_mask, 1);
  sum_hi = _mm512_add_epi64(sum_hi, carry_vec);
  res_lo = sum_lo;
  res_hi = sum_hi;
}

inline __m512i avx_umul64hi(__m512i a, __m512i b) {
  const __m512i a_hi = _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)0xB1);
  const __m512i b_hi = _mm512_shuffle_epi32(b, (_MM_PERM_ENUM)0xB1);
  const __m512i z_lo_lo = _mm512_mul_epu32(a, b);
  const __m512i z_lo_hi = _mm512_mul_epu32(a, b_hi);
  const __m512i z_hi_lo = _mm512_mul_epu32(a_hi, b);
  const __m512i z_hi_hi = _mm512_mul_epu32(a_hi, b_hi);
  const __m512i z_lo_lo_shift = _mm512_srli_epi64(z_lo_lo, 32);
  const __m512i sum_tmp = _mm512_add_epi64(z_lo_hi, z_lo_lo_shift);
  const __m512i sum_lo = _mm512_and_si512(sum_tmp, lo_mask);
  const __m512i sum_mid = _mm512_srli_epi64(sum_tmp, 32);
  const __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
  const __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
  const __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
  return _mm512_add_epi64(sum_hi, sum_mid2_hi);
}

static inline __attribute__((always_inline)) __m512i barret_reduction_64_64_avx512(
    const __m512i& x,
    const __m512i& modulus,
    const __m512i& mu,
    unsigned shift_bits) {
  const unsigned shift = shift_bits - 64;
  __m512i q = avx_umul64hi(x, mu);
  q = _mm512_srli_epi64(q, shift);
  const __m512i prod = _mm512_mullo_epi64(q, modulus);
  const __m512i r = _mm512_sub_epi64(x, prod);
  const __mmask8 lt_mask = _mm512_cmp_epu64_mask(r, modulus, _MM_CMPINT_LT);
  const __m512i r_minus_mod = _mm512_sub_epi64(r, modulus);
  return _mm512_mask_blend_epi64(lt_mask, r_minus_mod, r);
}

static inline __m512i barret_reduction_128_64_avx512(
    const __m512i& x_lo,
    const __m512i& x_hi,
    const __m512i& p,
    const __m512i& mu,
    unsigned k) {
  const __m512i m_lo_hi = avx_umul64hi(x_lo, mu);
  const __m512i m_hi_hi = avx_umul64hi(x_hi, mu);
  const __m512i m_hi_lo = _mm512_mullo_epi64(x_hi, mu);
  const __m512i sum_lo = _mm512_add_epi64(m_hi_lo, m_lo_hi);
  const __mmask8 carry_mask = _mm512_cmp_epu64_mask(sum_lo, m_hi_lo, _MM_CMPINT_LT);
  const __m512i one = _mm512_set1_epi64(1);
  const __m512i sum_hi = _mm512_mask_add_epi64(m_hi_hi, carry_mask, m_hi_hi, one);

  const unsigned shift_amount = k - 64;
  const __m512i shifted_lo = _mm512_srli_epi64(sum_lo, shift_amount);
  const __m512i shifted_hi = _mm512_slli_epi64(sum_hi, 64 - shift_amount);
  const __m512i q = _mm512_or_si512(shifted_lo, shifted_hi);
  const __m512i prod_lo = _mm512_mullo_epi64(q, p);
  const __m512i r = _mm512_sub_epi64(x_lo, prod_lo);
  const __mmask8 ge_mask = _mm512_cmp_epu64_mask(r, p, _MM_CMPINT_GE);
  return _mm512_mask_sub_epi64(r, ge_mask, r, p);
}

static inline __attribute__((always_inline)) __m512i
mul_and_reduce_shoup_avx512_full(
    const __m512i& a,
    const __m512i& b,
    const __m512i& bInv,
    const __m512i& p) {
  const __m512i q = avx_umul64hi(a, bInv);
  const __m512i prod_ab = _mm512_mullo_epi64(a, b);
  const __m512i prod_qp = _mm512_mullo_epi64(q, p);
  return _mm512_sub_epi64(prod_ab, prod_qp);
}
#endif

static inline uint64_t barret_reduction_128_64(
    __uint128_t x,
    uint64_t p,
    uint64_t mu,
    unsigned k // must satisfy 64 < k < 128
) {
  // split x
  uint64_t lo = uint64_t(x);
  uint64_t hi = uint64_t(x >> 64);

  // two 64*64 -> 128 multiplies
  __uint128_t m_lo = __uint128_t(lo) * mu;
  __uint128_t m_hi = __uint128_t(hi) * mu;

  // carry‐propagate the upper half of m_lo into m_hi
  uint64_t mid = uint64_t(m_lo >> 64);
  __uint128_t sum = m_hi + mid;

  // shift down by (k−64) to get the approximate quotient
  uint64_t q = uint64_t(sum >> (k - 64));

  // subtract and correct
  __uint128_t prod = __uint128_t(q) * p;
  uint64_t r = uint64_t(x - prod);
  return (r >= p) ? r - p : r;
}


inline uint64_t mul_mod_barrett(
    uint64_t a,
    uint64_t b,
    uint64_t p,
    uint64_t mu, // ratio
    unsigned shift_bits // k
) {
  __uint128_t prod = (__uint128_t)a * b;
  return barret_reduction_128_64(prod, p, mu, shift_bits);
}


#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
inline uint64_t umul64hi(uint64_t a, uint64_t b) {
  uint64_t hi;
  _mulx_u64(a, b, &hi);
  return hi;
}
#else
inline uint64_t umul64hi(uint64_t a, uint64_t b) {
  return uint64_t((unsigned __int128)a * b >> 64);
}
#endif

inline uint64_t barret_reduction_64_64(
    uint64_t x,
    uint64_t modulus,
    uint64_t mu,
    unsigned shift_bits) {
  // q = floor(x * mu / 2^shift_bits)
  uint64_t q = umul64hi(x, mu) >> (shift_bits - 64);
  uint64_t r = x - q * modulus;
  return (r < modulus) ? r : (r - modulus);
}

static inline uint64_t mul_and_reduce_shoup(
    uint64_t a,
    uint64_t b,
    uint64_t bInv,
    uint64_t p) {
  uint64_t q = umul64hi(a, bInv);
  return a * b - q * p;
}

} // namespace fhe
