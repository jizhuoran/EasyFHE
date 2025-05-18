#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>
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
    int64_t level,
    int64_t param_degree,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace,
    Tensor& res) {
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - level;
  const int length = (curr_limbs + sizeP);
  int gap = level - curr_limbs;
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
//  omp_set_num_threads(max_threads);
  for (uint32_t j = 0; j < beta; ++j) {
    const uint64_t* d2_ptr = modup_out_ptr + j * param_degree * length;
    const uint64_t* d_ax_ptr = ax_ptr + j * param_degree * sizeQP;
    const uint64_t* d_bx_ptr = bx_ptr + j * param_degree * sizeQP;

//#pragma omp parallel for schedule(static) num_threads(max_threads)
    for (uint64_t idx = 0; idx < length; ++idx) {
        const int prime_idx = (idx < curr_limbs) ? 0 : gap;
        for (uint64_t k = 0; k < param_degree; ++k) {
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
          const int prime_idx1 = idx + ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
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
}

Tensor innerproduct_cpu(
    const Tensor& res,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
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
  innerproduct_template(
      in,
      bx,
      ax,
      curr_limbs,
      alpha,
      L,
      N,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      out);
  return out;
}

} // namespace at::native
