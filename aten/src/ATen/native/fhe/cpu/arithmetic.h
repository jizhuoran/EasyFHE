#pragma once

#ifdef USE_AVX512
#include <immintrin.h>
#endif
#include <cstdint>
#include "ATen/native/fhe/cpu/Utils.h"
namespace fhe {

#ifdef USE_AVX512
static inline __attribute__((always_inline)) __m512i
neg_mod_avx512(const __m512i& x, const __m512i&, const __m512i& mod) {
  const __mmask8 zero_mask = _mm512_cmp_epu64_mask(x, _mm512_setzero_si512(), _MM_CMPINT_EQ);
  const __m512i neg_result = _mm512_sub_epi64(mod, x);
  return _mm512_mask_mov_epi64(_mm512_setzero_si512(), ~zero_mask, neg_result);
}

static inline __attribute__((always_inline)) __m512i
add_mod_avx512(const __m512i& a, const __m512i& b, const __m512i& mod) {
  const __m512i res = _mm512_add_epi64(a, b);
  const __mmask8 reduce_mask = _mm512_cmp_epu64_mask(res, mod, _MM_CMPINT_GE);
  return _mm512_mask_sub_epi64(res, reduce_mask, res, mod);
}

static inline __attribute__((always_inline)) __m512i
sub_mod_avx512(const __m512i& a, const __m512i& b, const __m512i& mod) {
  const __mmask8 borrow_mask = _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LT);
  const __m512i adjusted_a = _mm512_mask_add_epi64(a, borrow_mask, a, mod);
  return _mm512_sub_epi64(adjusted_a, b);
}

static inline __attribute__((always_inline)) __m512i mul_mod_avx512(
    const __m512i& a,
    const __m512i& b,
    const __m512i& mod,
    const __m512i& mu0,
    const __m512i& mu1) {
  const __m512i zero = _mm512_setzero_si512();
  const __m512i p_lo = _mm512_mullo_epi64(a, b);
  const __m512i p_hi = avx_umul64hi(a, b);

  const __m512i A = avx_umul64hi(p_lo, mu0);
  const __m512i B_lo = _mm512_mullo_epi64(p_lo, mu1);
  const __m512i B_hi = avx_umul64hi(p_lo, mu1);
  const __m512i C_lo = _mm512_mullo_epi64(p_hi, mu0);
  const __m512i C_hi = avx_umul64hi(p_hi, mu0);

  const __m512i s1 = _mm512_add_epi64(A, B_lo);
  const __mmask8 c1m = _mm512_cmp_epu64_mask(s1, A, _MM_CMPINT_LT);
  const __m512i s2 = _mm512_add_epi64(s1, C_lo);
  const __mmask8 c2m = _mm512_cmp_epu64_mask(s2, s1, _MM_CMPINT_LT);
  const __mmask8 carry_mask = c1m | c2m;
  const __m512i carry = _mm512_maskz_set1_epi64(carry_mask, 1);

  __m512i t_hi = _mm512_add_epi64(B_hi, C_hi);
  t_hi = _mm512_add_epi64(t_hi, carry);
  const __m512i D_hi = _mm512_mullo_epi64(p_hi, mu1);
  const __m512i q = _mm512_add_epi64(t_hi, D_hi);

  const __m512i qm_lo = _mm512_mullo_epi64(q, mod);
  const __m512i qm_hi = avx_umul64hi(q, mod);
  const __m512i r_lo = _mm512_sub_epi64(p_lo, qm_lo);
  const __mmask8 borrow0 = _mm512_cmp_epu64_mask(p_lo, qm_lo, _MM_CMPINT_LT);
  __m512i r_hi = _mm512_sub_epi64(p_hi, qm_hi);
  r_hi = _mm512_sub_epi64(r_hi, _mm512_maskz_set1_epi64(borrow0, 1));

  const __mmask8 ge_hi = _mm512_cmp_epu64_mask(r_hi, zero, _MM_CMPINT_NE);
  const __mmask8 ge_lo = _mm512_cmp_epu64_mask(r_lo, mod, _MM_CMPINT_GE);
  const __mmask8 ge = ge_hi | ge_lo;
  return _mm512_mask_sub_epi64(r_lo, ge, r_lo, mod);
}
#endif

static inline __attribute__((always_inline)) uint64_t
neg_mod(uint64_t x, uint64_t null, uint64_t mod) {
  return x == 0 ? 0 : mod - x;
}

static inline __attribute__((always_inline)) uint64_t
add_mod(uint64_t a, uint64_t b, uint64_t mod) {
  uint64_t res = a + b;
  return res >= mod ? res - mod : res;
}

static inline __attribute__((always_inline)) uint64_t
sub_mod(uint64_t a, uint64_t b, uint64_t mod) {
  return a >= b ? a - b : a + mod - b;
}

static inline __attribute__((always_inline)) uint64_t
mul_mod(uint64_t a, uint64_t b, uint64_t mod, uint64_t mu0, uint64_t mu1) {
  unsigned __int128 p = (unsigned __int128)a * b;
  uint64_t p_lo = (uint64_t)p;
  uint64_t p_hi = p >> 64;

  unsigned __int128 t = ((unsigned __int128)p_lo * mu0) >> 64;
  t += (unsigned __int128)p_lo * mu1;
  t += (unsigned __int128)p_hi * mu0;

  uint64_t q = (uint64_t)(t >> 64) + (uint64_t)((unsigned __int128)p_hi * mu1);

  unsigned __int128 r = p - (unsigned __int128)q * mod;

  return (uint64_t)(r >= mod ? r - mod : r);
}

} // namespace fhe

namespace at::native {


void vadd_mod(
  const size_t N,
  int64_t l,
  uint64_t* c,
  const uint64_t* a,
  const uint64_t* b,
  const uint64_t* mod);

void vsub_mod(
  const size_t N,
  int64_t l,
  uint64_t* c,
  const uint64_t* a,
  const uint64_t* b,
  const uint64_t* mod);

void vneg_mod(
  const size_t N,
  int64_t l,
  uint64_t* c,
  const uint64_t* a,
  const uint64_t* b,
  const uint64_t* mod);

} // namespace at::native
