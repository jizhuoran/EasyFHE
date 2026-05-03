#pragma once
#include <ATen/Tensor.h>
#include <cstdint>
#ifdef USE_AVX512
#include <immintrin.h>
#endif
#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/Utils.h"
namespace fhe {

#ifdef USE_AVX512
static inline __attribute__((always_inline)) void butt_ntt_local_avx512(
    __m512i& vec_a,
    __m512i& vec_b,
    const __m512i& vec_w,
    const __m512i& vec_w_,
    const __m512i& vec_p) {
  const __m512i vec_u = mul_and_reduce_shoup_avx512_full(vec_b, vec_w, vec_w_, vec_p);
  const __m512i vec_two_p = _mm512_slli_epi64(vec_p, 1);
  const __m512i tmp = _mm512_sub_epi64(vec_two_p, vec_u);
  const __mmask8 ge_2p = _mm512_cmp_epu64_mask(vec_a, vec_two_p, _MM_CMPINT_GE);
  vec_a = _mm512_mask_sub_epi64(vec_a, ge_2p, vec_a, vec_two_p);
  vec_b = _mm512_add_epi64(vec_a, tmp);
  vec_a = _mm512_add_epi64(vec_a, vec_u);
}

static inline __attribute__((always_inline)) void butt_intt_local_avx512(
    __m512i& vec_a,
    __m512i& vec_b,
    const __m512i& vec_w,
    const __m512i& vec_w_,
    const __m512i& vec_p) {
  const __m512i vec_two_p = _mm512_slli_epi64(vec_p, 1);
  __m512i vec_t = _mm512_sub_epi64(vec_two_p, vec_b);
  vec_t = _mm512_add_epi64(vec_t, vec_a);
  __m512i vec_new_a = _mm512_add_epi64(vec_a, vec_b);
  const __mmask8 ge_2p = _mm512_cmp_epu64_mask(vec_new_a, vec_two_p, _MM_CMPINT_GE);
  vec_new_a = _mm512_mask_sub_epi64(vec_new_a, ge_2p, vec_new_a, vec_two_p);

  const __m512i one = _mm512_set1_epi64(1);
  const __mmask8 odd_mask = _mm512_test_epi64_mask(vec_t, one);
  vec_new_a = _mm512_mask_add_epi64(vec_new_a, odd_mask, vec_new_a, vec_p);
  vec_a = _mm512_srli_epi64(vec_new_a, 1);
  vec_b = mul_and_reduce_shoup_avx512_full(vec_t, vec_w, vec_w_, vec_p);
}
#endif

static inline __attribute__((always_inline)) void butt_intt_local(
    uint64_t& x,
    uint64_t& y,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t& p) {
  const uint64_t two_p = 2 * p;
  const uint64_t T = two_p - y + x;
  uint64_t new_x = x + y;
  if (new_x >= two_p)
    new_x -= two_p;
  if (T & 1)
    new_x += p;
  x = (new_x >> 1);
  y = mul_and_reduce_shoup(T, w, w_, p);
}

static inline __attribute__((always_inline)) void butt_ntt_local(
    uint64_t& a,
    uint64_t& b,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t p) {
  uint64_t two_p = 2 * p;
  uint64_t U = mul_and_reduce_shoup(b, w, w_, p);
  if (a >= two_p)
    a -= two_p;
  b = a + (two_p - U);
  a += U;
}
} // namespace fhe
