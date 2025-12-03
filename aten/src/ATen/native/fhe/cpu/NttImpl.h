#pragma once
#include <ATen/Tensor.h>
#include <immintrin.h>
#include <cstdint>
#include "ATen/native/fhe/cpu/Utils.h"
namespace fhe {
#ifdef USE_AVX512
static inline __attribute__((always_inline)) void butt_ntt_local_avx512(__m512i& vec_a,
                                                                        __m512i& vec_b,
                                                                        const __m512i& vec_w,
                                                                        const __m512i& vec_w_,
                                                                        const __m512i& vec_p) {
  __m512i vec_U = mul_and_reduce_shoup_avx512_full(vec_b, vec_w, vec_w_, vec_p);

  __m512i vec_two_p = _mm512_slli_epi64(vec_p, 1);
  __m512i temp = _mm512_sub_epi64(vec_two_p, vec_U);
  vec_a = _mm512_min_epu64(vec_a, _mm512_sub_epi64(vec_a, vec_two_p));
  vec_b = _mm512_add_epi64(vec_a, temp);
  vec_a = _mm512_add_epi64(vec_a, vec_U);
}
static inline __attribute__((always_inline)) void butt_intt_local_avx512(__m512i& vec_a,
                                                                         __m512i& vec_b,
                                                                         const __m512i& vec_w,
                                                                         const __m512i& vec_w_,
                                                                         const __m512i& vec_p) {
  // tseting new feature
  __m512i vec_two_p = _mm512_slli_epi64(vec_p, 1);
  __m512i vec_T = _mm512_sub_epi64(vec_two_p, vec_b);
  vec_T = _mm512_add_epi64(vec_T, vec_a);
  __m512i vec_new_a = _mm512_add_epi64(vec_a, vec_b);
  __mmask8 mask_ge_2p = _mm512_cmpge_epu64_mask(vec_new_a, vec_two_p);
  vec_new_a = _mm512_mask_sub_epi64(vec_new_a, mask_ge_2p, vec_new_a, vec_two_p);

  __m512i vec_one = _mm512_set1_epi64(1);
  __mmask8 mask_odd = _mm512_test_epi64_mask(vec_T, vec_one);
  vec_new_a = _mm512_mask_add_epi64(vec_new_a, mask_odd, vec_new_a, vec_p);
  vec_a = _mm512_srli_epi64(vec_new_a, 1);
  vec_b = mul_and_reduce_shoup_avx512_full(vec_T, vec_w, vec_w_, vec_p);
}
#endif
static inline __attribute__((always_inline)) void butt_intt_local(uint64_t& x,
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

static inline __attribute__((always_inline)) void butt_ntt_local(uint64_t& a,
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

namespace at::native {

void iNTT_impl(uint64_t* out_ptr,
               uint64_t* in_ptr,
               int64_t start_prime_idx,
               int64_t batch,
               int64_t curr_limbs,
               int64_t level,
               int64_t param_degree,
               const Tensor& inverse_power_of_roots_div_two,
               const Tensor& param_primes,
               const Tensor& inverse_scaled_power_of_roots_div_two);

void NTT_impl(uint64_t* inout_ptr,
              int64_t start_prime_idx,
              int64_t batch,
              int64_t param_degree,
              const Tensor& param_power_of_roots_shoup,
              const Tensor& param_primes,
              const Tensor& param_power_of_roots);

void NTT_except_some_range_impl(uint64_t* op_ptr,
                                int64_t start_prime_idx,
                                int64_t batch,
                                int64_t param_degree,
                                int64_t excluded_range_start,
                                int64_t excluded_range_size,
                                int64_t curr_limbs,
                                int64_t level,
                                const Tensor& param_power_of_roots_shoup,
                                const Tensor& param_primes,
                                const Tensor& param_power_of_roots);

} // namespace at::native
