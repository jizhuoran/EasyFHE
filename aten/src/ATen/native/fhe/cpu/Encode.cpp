#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <limits>
#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/Utils.h"
#define MAX_64BIT_VALUE 9223372036854775295LL

namespace fhe {
template <typename DDTYPE>
void new_fit_to_native_vector_host(
    DDTYPE* inverse,
    double scaling_factor,
    int64_t bigValueHf,
    uint64_t* native_vec,
    const uint64_t* native_modulus,
    const uint64_t* max_int_diffs_ptr,
    const uint64_t* barret_ratio_ptr,
    const uint64_t* barret_k_ptr,
    int64_t N,
    int64_t slots,
    int64_t gap,
    int64_t cur_limbs,
    bool is_ext,
    int64_t sizeP) {
  int total_limbs = cur_limbs + (is_ext ? sizeP : 0);
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int64_t l = 0; l < total_limbs; ++l) {
    uint64_t modulus = native_modulus[l];
    uint64_t diff = max_int_diffs_ptr[l];
    uint64_t ratio = barret_ratio_ptr[l];
    uint64_t k = barret_k_ptr[l];
#ifdef USE_AVX512
    const __m512i modulus_vec = _mm512_set1_epi64((long long)modulus);
    const __m512i ratio_vec = _mm512_set1_epi64((long long)ratio);
    unsigned shift_bits = static_cast<unsigned>(k);
    int64_t i = 0;
    for (; i + 7 < slots; i += 8) {
      int64_t re_int[8];
      int64_t im_int[8];
      uint64_t re_tmp[8];
      uint64_t im_tmp[8];
      // gather / convert scalars into small temp arrays
      for (int t = 0; t < 8; ++t) {
        int64_t re = static_cast<int64_t>(
            std::llround(inverse[2 * (i + t)] * scaling_factor));
        int64_t im = static_cast<int64_t>(
            std::llround(inverse[2 * (i + t) + 1] * scaling_factor));
        if (re < 0)
          re = MAX_64BIT_VALUE + re;
        if (im < 0)
          im = MAX_64BIT_VALUE + im;
        re_int[t] = re;
        im_int[t] = im;
        re_tmp[t] = static_cast<uint64_t>(re);
        im_tmp[t] = static_cast<uint64_t>(im);
      }

      // load into vectors and perform AVX512 Barrett reduction
      __m512i re_vec = _mm512_loadu_si512((const __m512i*)re_tmp);
      __m512i im_vec = _mm512_loadu_si512((const __m512i*)im_tmp);
      __m512i re_red = barret_reduction_64_64_avx512(
          re_vec, modulus_vec, ratio_vec, shift_bits);
      __m512i im_red = barret_reduction_64_64_avx512(
          im_vec, modulus_vec, ratio_vec, shift_bits);

      const __m512i bv_vec = _mm512_set1_epi64((long long)bigValueHf);
      __mmask8 re_mask = _mm512_cmpgt_epu64_mask(re_vec, bv_vec);
      __mmask8 im_mask = _mm512_cmpgt_epu64_mask(im_vec, bv_vec);

      // 只有被标记的 lane 需要做减 diff 的修正
      const __m512i diff_vec = _mm512_set1_epi64((long long)diff);
      __m512i re_corrected = sub_mod_avx512(re_red, diff_vec, modulus_vec);
      __m512i im_corrected = sub_mod_avx512(im_red, diff_vec, modulus_vec);

      __m512i re_final = _mm512_mask_blend_epi64(re_mask, re_red, re_corrected);
      __m512i im_final = _mm512_mask_blend_epi64(im_mask, im_red, im_corrected);

      uint64_t re_out[8];
      uint64_t im_out[8];
      _mm512_storeu_si512((__m512i*)re_out, re_final);
      _mm512_storeu_si512((__m512i*)im_out, im_final);

      for (int t = 0; t < 8; ++t) {
        int64_t s = i + t;
        native_vec[l * N + gap * s] = re_out[t];
        native_vec[l * N + gap * (s + slots)] = im_out[t];
      }
    }
    // tail scalar for remaining slots
    for (; i < slots; ++i) {
      int64_t re =
          static_cast<int64_t>(std::llround(inverse[2 * i] * scaling_factor));
      int64_t im = static_cast<int64_t>(
          std::llround(inverse[2 * i + 1] * scaling_factor));
      if (re < 0)
        re = MAX_64BIT_VALUE + re;
      if (im < 0)
        im = MAX_64BIT_VALUE + im;
      uint64_t re_u = static_cast<uint64_t>(re);
      uint64_t im_u = static_cast<uint64_t>(im);
      re_u = barret_reduction_64_64(re_u, modulus, ratio, k);
      im_u = barret_reduction_64_64(im_u, modulus, ratio, k);
      if (re > bigValueHf) {
        re_u = sub_mod(re_u, diff, modulus);
      }
      if (im > bigValueHf) {
        im_u = sub_mod(im_u, diff, modulus);
      }
      native_vec[l * N + gap * i] = re_u;
      native_vec[l * N + gap * (i + slots)] = im_u;
    }
#else
    for (int64_t i = 0; i < slots; ++i) {
      int64_t re =
          static_cast<int64_t>(std::llround(inverse[2 * i] * scaling_factor));
      int64_t im = static_cast<int64_t>(
          std::llround(inverse[2 * i + 1] * scaling_factor));
      if (re < 0)
        re = MAX_64BIT_VALUE + re;
      if (im < 0)
        im = MAX_64BIT_VALUE + im;
      uint64_t re_u = static_cast<uint64_t>(re);
      uint64_t im_u = static_cast<uint64_t>(im);
      re_u = barret_reduction_64_64(re_u, modulus, ratio, k);
      im_u = barret_reduction_64_64(im_u, modulus, ratio, k);

      if (re > bigValueHf) {
        re_u = sub_mod(re_u, diff, modulus);
      }
      if (im > bigValueHf) {
        im_u = sub_mod(im_u, diff, modulus);
      }
      native_vec[l * N + gap * i] = re_u;
      native_vec[l * N + gap * (i + slots)] = im_u;
    }
#endif
  }
}

} // namespace fhe
namespace at::native {
static void encode_template_cpu(
    Tensor& res,
    const Tensor& input,
    const Tensor& max_int_diffs,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    double scaling_factor,
    bool is_ext,
    int64_t sizeP,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  auto elements_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto max_int_diffs_ptr =
      reinterpret_cast<uint64_t*>(max_int_diffs.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  const auto gap = N / (slots * 2);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "encode_template_cpu", [&]() {
    auto input_ptr = input.data_ptr<scalar_t>();
    fhe::new_fit_to_native_vector_host<scalar_t>(
        input_ptr,
        scaling_factor,
        static_cast<int64_t>(MAX_64BIT_VALUE >> 1),
        elements_ptr,
        primes_ptr,
        max_int_diffs_ptr,
        barret_ratio_ptr,
        barret_k_ptr,
        N,
        slots,
        gap,
        cur_limbs,
        is_ext,
        sizeP);
  });
  NTT_impl(
      elements_ptr,
      0,
      cur_limbs,
      N,
      power_of_roots_shoup,
      primes,
      power_of_roots);
  if (is_ext) {
    const auto L = power_of_roots.numel() / N - sizeP;
    auto offset = L * N;
    Tensor power_of_roots_shoup_offset = power_of_roots_shoup.narrow(
        0, offset, power_of_roots_shoup.numel() - offset);
    Tensor power_of_roots_offset =
        power_of_roots.narrow(0, offset, power_of_roots.numel() - offset);
    Tensor primes_offset =
        primes.narrow(0, cur_limbs, primes.numel() - cur_limbs);
    NTT_impl(
        elements_ptr + cur_limbs * N,
        0,
        sizeP,
        N,
        power_of_roots_shoup_offset,
        primes_offset,
        power_of_roots_offset);
  }
}

Tensor encode_cpu(
    const Tensor& inverse_internal,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    double scaling_factor,
    bool is_ext,
    int64_t sizeP,
    const Tensor& primes,
    const Tensor& max_int_diffs,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  Tensor out =
      at::zeros({cur_limbs + (is_ext ? sizeP : 0), N}, primes.options());
  encode_template_cpu(
      out,
      inverse_internal,
      max_int_diffs,
      N,
      cur_limbs,
      slots,
      scaling_factor,
      is_ext,
      sizeP,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots);
  return out;
}
} // namespace at::native
