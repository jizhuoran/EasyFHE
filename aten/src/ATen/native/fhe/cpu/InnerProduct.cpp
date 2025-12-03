#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>
#include <immintrin.h>
#include <omp.h>
#include "ATen/native/fhe/cpu/Utils.h"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

static void innerproduct_template(
    const Tensor& modup_out,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t level,
    int64_t param_degree,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace,
    Tensor& res,
    Tensor& accum_ax_lo_tensor,
    Tensor& accum_ax_hi_tensor,
    Tensor& accum_bx_lo_tensor,
    Tensor& accum_bx_hi_tensor) {
#ifdef USE_AVX512
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - level;
  const int length = (curr_limbs + sizeP);
  int prime_gap = level - curr_limbs;
  int swk_gap = special_mod_start - curr_limbs;
  __uint128_t* accum_bx_ptr =
      reinterpret_cast<__uint128_t*>(workspace.data_ptr<uint64_t>());
  __uint128_t* accum_ax_ptr = accum_bx_ptr + modup_out.size(-1);
  auto modup_out_ptr =
      reinterpret_cast<uint64_t*>(modup_out.data_ptr<uint64_t>());
  auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
  auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
  auto res_bx_ptr = reinterpret_cast<uint64_t*>(res[0].data_ptr<uint64_t>());
  auto res_ax_ptr = reinterpret_cast<uint64_t*>(res[1].data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto accum_ax_lo =
      reinterpret_cast<uint64_t*>(accum_ax_lo_tensor.data_ptr<uint64_t>());
  auto accum_ax_hi =
      reinterpret_cast<uint64_t*>(accum_ax_hi_tensor.data_ptr<uint64_t>());
  auto accum_bx_lo =
      reinterpret_cast<uint64_t*>(accum_bx_lo_tensor.data_ptr<uint64_t>());
  auto accum_bx_hi =
      reinterpret_cast<uint64_t*>(accum_bx_hi_tensor.data_ptr<uint64_t>());
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (uint64_t idx = 0; idx < length; ++idx) {
    const int swk_idx = (idx < curr_limbs) ? 0 : swk_gap;
    for (uint64_t k = 0; k < param_degree; k += 8) {
      auto i = idx * param_degree + k;

      __m512i reg_acc_ax_lo = _mm512_setzero_si512();
      __m512i reg_acc_ax_hi = _mm512_setzero_si512();
      __m512i reg_acc_bx_lo = _mm512_setzero_si512();
      __m512i reg_acc_bx_hi = _mm512_setzero_si512();

      for (uint32_t j = 0; j < beta; ++j) {
        const uint64_t* d2_ptr = modup_out_ptr + j * param_degree * length;
        const uint64_t* d_ax_ptr = ax_ptr + j * param_degree * sizeQP;
        const uint64_t* d_bx_ptr = bx_ptr + j * param_degree * sizeQP;

        __m512i v_op1 = _mm512_loadu_si512(&d2_ptr[i]);
        __m512i v_op2_ax =
            _mm512_loadu_si512(&d_ax_ptr[i + param_degree * swk_idx]);
        __m512i v_op2_bx =
            _mm512_loadu_si512(&d_bx_ptr[i + param_degree * swk_idx]);

        __m512i mul_ax_lo = _mm512_mullo_epi64(v_op1, v_op2_ax);
        __m512i mul_ax_hi = fhe::avx_umul64hi(v_op1, v_op2_ax);
        __m512i mul_bx_lo = _mm512_mullo_epi64(v_op1, v_op2_bx);
        __m512i mul_bx_hi = fhe::avx_umul64hi(v_op1, v_op2_bx);

        fhe::avx512_add_u128(
            reg_acc_ax_lo,
            reg_acc_ax_hi,
            mul_ax_lo,
            mul_ax_hi,
            reg_acc_ax_lo,
            reg_acc_ax_hi);
        fhe::avx512_add_u128(
            reg_acc_bx_lo,
            reg_acc_bx_hi,
            mul_bx_lo,
            mul_bx_hi,
            reg_acc_bx_lo,
            reg_acc_bx_hi);
      }

      _mm512_storeu_si512(&accum_ax_lo[i], reg_acc_ax_lo);
      _mm512_storeu_si512(&accum_ax_hi[i], reg_acc_ax_hi);
      _mm512_storeu_si512(&accum_bx_lo[i], reg_acc_bx_lo);
      _mm512_storeu_si512(&accum_bx_hi[i], reg_acc_bx_hi);

      if (beta > 0) {
        const int prime_idx =
            idx + ((idx >= 0 && idx < curr_limbs) ? 0 : prime_gap);
        const auto prime = primes_ptr[prime_idx];
        const auto barret_ratio = barret_ratio_ptr[prime_idx];
        const auto barret_k = barret_k_ptr[prime_idx];

        __m512i prime_vec = _mm512_set1_epi64(prime);
        __m512i mu_vec = _mm512_set1_epi64(barret_ratio);

        __m512i res_ax = fhe::barret_reduction_128_64_avx512(
            reg_acc_ax_lo, reg_acc_ax_hi, prime_vec, mu_vec, barret_k);
        __m512i res_bx = fhe::barret_reduction_128_64_avx512(
            reg_acc_bx_lo, reg_acc_bx_hi, prime_vec, mu_vec, barret_k);

        _mm512_storeu_si512((__m512i*)&res_ax_ptr[i], res_ax);
        _mm512_storeu_si512((__m512i*)&res_bx_ptr[i], res_bx);
      }
    }
  }
#else
  std::cout << "AVX512 not enabled!" << std::endl;
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - level;
  const int length = (curr_limbs + sizeP);
  int gap = special_mod_start - curr_limbs;
  __uint128_t* accum_bx_ptr =
      reinterpret_cast<__uint128_t*>(workspace.data_ptr<uint64_t>());
  __uint128_t* accum_ax_ptr = accum_bx_ptr + modup_out.size(-1);
  auto modup_out_ptr =
      reinterpret_cast<uint64_t*>(modup_out.data_ptr<uint64_t>());
  auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
  auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
  auto res_bx_ptr = reinterpret_cast<uint64_t*>(res[0].data_ptr<uint64_t>());
  auto res_ax_ptr = reinterpret_cast<uint64_t*>(res[1].data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  for (uint32_t j = 0; j < beta; ++j) {
    const uint64_t* d2_ptr = modup_out_ptr + j * param_degree * length;
    const uint64_t* d_ax_ptr = ax_ptr + j * param_degree * sizeQP;
    const uint64_t* d_bx_ptr = bx_ptr + j * param_degree * sizeQP;

#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads)
    for (uint64_t idx = 0; idx < length; ++idx) {
      for (uint64_t k = 0; k < param_degree; ++k) {
        const int prime_idx = (idx < curr_limbs) ? 0 : gap;
        auto i = idx * param_degree + k;
        const uint64_t op1 = d2_ptr[i];
        const uint64_t op2_ax = d_ax_ptr[i + param_degree * prime_idx];
        const uint64_t op2_bx = d_bx_ptr[i + param_degree * prime_idx];
        const auto mul_ax = static_cast<__uint128_t>(op1) * op2_ax;
        const auto mul_bx = static_cast<__uint128_t>(op1) * op2_bx;
        if (j == 0) {
          accum_ax_ptr[i] = mul_ax;
          accum_bx_ptr[i] = mul_bx;
        } else {
          accum_ax_ptr[i] += mul_ax;
          accum_bx_ptr[i] += mul_bx;
        }
        if (j == beta - 1) {
          const int prime_idx1 =
              idx + ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
          const auto prime = primes_ptr[prime_idx1];
          const auto barret_ratio = barret_ratio_ptr[prime_idx1];
          const auto barret_k = barret_k_ptr[prime_idx1];
          const auto res_ax = fhe::barret_reduction_128_64(
              accum_ax_ptr[i], prime, barret_ratio, barret_k);
          res_ax_ptr[i] = res_ax;
          const auto res_bx = fhe::barret_reduction_128_64(
              accum_bx_ptr[i], prime, barret_ratio, barret_k);
          res_bx_ptr[i] = res_bx;
        }
      }
    }
  }
#endif
}

Tensor innerproduct_cpu(
    const Tensor& res,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  Tensor out = at::empty_like(res);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  out.resize_({2, (curr_limbs + sizeP) * N});
  int64_t length = (curr_limbs + sizeP);
  Tensor accum_ax_lo = at::zeros({length * N}, at::kUInt64);
  Tensor accum_ax_hi = at::zeros({length * N}, at::kUInt64);
  Tensor accum_bx_lo = at::zeros({length * N}, at::kUInt64);
  Tensor accum_bx_hi = at::zeros({length * N}, at::kUInt64);
  innerproduct_template(
      in,
      bx,
      ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      N,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      out,
      accum_ax_lo,
      accum_ax_hi,
      accum_bx_lo,
      accum_bx_hi);
  return out;
}

} // namespace at::native
