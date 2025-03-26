#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cpu/Utils.h"

namespace fhe {

template <bool Accum>
void mult_(
    const uint64_t* modup_out,
    const uint64_t* eval_poly_ax,
    const uint64_t* eval_poly_bx,
    const int degree,
    const int length,
    uint128_t* accum_ptr_ax,
    uint128_t* accum_ptr_bx,
    int curr_limbs,
    int gap) {
  STRIDED_LOOP_START(degree * length, i);
  const uint64_t op1 = modup_out[i];
  const int idx = i / degree;
  const int prime_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : gap);

  const uint64_t op2_ax = eval_poly_ax[i + degree * prime_idx];
  const uint64_t op2_bx = eval_poly_bx[i + degree * prime_idx];
  const auto mul_ax = mult_64_64_128(op1, op2_ax);
  const auto mul_bx = mult_64_64_128(op1, op2_bx);
  if (Accum) {
    accum_ptr_ax[i] += mul_ax;
    accum_ptr_bx[i] += mul_bx;
  } else {
    accum_ptr_ax[i] = mul_ax;
    accum_ptr_bx[i] = mul_bx;
  }
  STRIDED_LOOP_END;
}

void Reduce(
    const uint128_t* accum,
    const int degree,
    const int length,
    const int curr_limbs,
    const int gap,
    const uint64_t* primes,
    const uint64_t* barret_ks,
    const uint64_t* barret_ratios,
    uint64_t* res) {
  STRIDED_LOOP_START(degree * length, i);
  const int idx = i / degree;
  const int prime_idx = idx + ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barret_ratios[prime_idx];
  const auto barret_k = barret_ks[prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum[i], prime, barret_ratio, barret_k);
  res[i] = res_ax;
  STRIDED_LOOP_END;
}
} // namespace fhe

namespace at::native {
// static void innerproduct_template(
//     const Tensor& modup_out,
//     const Tensor& bx,
//     const Tensor& ax,
//     int64_t curr_limbs,
//     int64_t alpha,
//     int64_t level,
//     int64_t param_degree,
//     const Tensor& primes,
//     const Tensor& barret_ratio,
//     const Tensor& barret_k,
//     const Tensor& workspace,
//     Tensor& res) {
//   const int total_length = modup_out.size(-1) / param_degree;
//   const int beta = total_length / (curr_limbs + alpha);
//   const int length = (curr_limbs + alpha);
//   const int mult_length = (level + alpha);
//   int gap = level - curr_limbs;

//   __uint128_t* accum_bx_ptr =
//       reinterpret_cast<__uint128_t*>(workspace.data_ptr<uint64_t>());
//   __uint128_t* accum_ax_ptr = accum_bx_ptr + modup_out.size(-1);

//   AT_DISPATCH_V2(
//       ax.scalar_type(),
//       "inner_product_impl",
//       AT_WRAP([&]() {
//         auto modup_out_ptr =
//             reinterpret_cast<uint64_t*>(modup_out.data_ptr<uint64_t>());
//         auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
//         auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
//         auto res_bx_ptr =
//             reinterpret_cast<uint64_t*>(res[0].data_ptr<uint64_t>());
//         auto res_ax_ptr =
//             reinterpret_cast<uint64_t*>(res[1].data_ptr<uint64_t>());
//         auto primes_ptr =
//             reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
//         auto barret_ratio_ptr =
//             reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
//         auto barret_k_ptr =
//             reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
//         const int gridDim = 1024;
//         const int blockDim = 256;
//         fhe::mult_<false>(
//             modup_out_ptr,
//             ax_ptr,
//             bx_ptr,
//             param_degree,
//             length,
//             accum_ax_ptr,
//             accum_bx_ptr,
//             curr_limbs,
//             gap);
//         for (int i = 1; i < beta; i++) {
//           auto d2_ptr = modup_out_ptr + i * param_degree * length;
//           auto d_ax_ptr = ax_ptr + i * param_degree * mult_length;
//           auto d_bx_ptr = bx_ptr + i * param_degree * mult_length;
//           fhe::mult_<true>(
//               d2_ptr,
//               d_ax_ptr,
//               d_bx_ptr,
//               param_degree,
//               length,
//               accum_ax_ptr,
//               accum_bx_ptr,
//               curr_limbs,
//               gap);
//         }
//         fhe::Reduce(
//             accum_ax_ptr,
//             param_degree,
//             length,
//             curr_limbs,
//             gap,
//             primes_ptr,
//             barret_k_ptr,
//             barret_ratio_ptr,
//             res_ax_ptr);
//         fhe::Reduce(
//             accum_bx_ptr,
//             param_degree,
//             length,
//             curr_limbs,
//             gap,
//             primes_ptr,
//             barret_k_ptr,
//             barret_ratio_ptr,
//             res_bx_ptr);
//       }),
//       kUInt64);
// }
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
  const int total_length = modup_out.size(-1) / param_degree;
  const int beta = total_length / (curr_limbs + alpha);
  const int length = (curr_limbs + alpha);
  const int mult_length = (level + alpha);
  int gap = level - curr_limbs;

  __uint128_t* accum_bx_ptr =
      reinterpret_cast<__uint128_t*>(workspace.data_ptr<uint64_t>());
  __uint128_t* accum_ax_ptr = accum_bx_ptr + modup_out.size(-1);

  AT_DISPATCH_V2(
      ax.scalar_type(),
      "inner_product_impl",
      AT_WRAP([&]() {
        auto modup_out_ptr =
            reinterpret_cast<uint64_t*>(modup_out.data_ptr<uint64_t>());
        auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
        auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
        auto res_bx_ptr =
            reinterpret_cast<uint64_t*>(res[0].data_ptr<uint64_t>());
        auto res_ax_ptr =
            reinterpret_cast<uint64_t*>(res[1].data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());

        //     for (uint64_t i = 0; i < param_degree*length; i++) {
        //         const uint64_t op1 = modup_out_ptr[i];
        //         const int idx = i / param_degree;
        //         const int prime_idx = ((idx >= 0 && idx < curr_limbs) ? 0 :
        //         gap); const uint64_t op2_ax = ax_ptr[i + param_degree *
        //         prime_idx]; const uint64_t op2_bx = bx_ptr[i + param_degree *
        //         prime_idx]; const auto mul_ax = fhe::mult_64_64_128(op1,
        //         op2_ax); const auto mul_bx = fhe::mult_64_64_128(op1,
        //         op2_bx); accum_ax_ptr[i] = mul_ax; accum_bx_ptr[i] = mul_bx;
        // }
        for (uint32_t j = 0; j < beta; j++) {
          auto d2_ptr = modup_out_ptr + j * param_degree * length;
          auto d_ax_ptr = ax_ptr + j * param_degree * mult_length;
          auto d_bx_ptr = bx_ptr + j * param_degree * mult_length;
          for (uint64_t i = 0; i < param_degree * length; i++) {
            const uint64_t op1 = d2_ptr[i];
            const int idx = i / param_degree;
            const int prime_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
            const uint64_t op2_ax = d_ax_ptr[i + param_degree * prime_idx];
            const uint64_t op2_bx = d_bx_ptr[i + param_degree * prime_idx];
            const auto mul_ax = fhe::mult_64_64_128(op1, op2_ax);
            const auto mul_bx = fhe::mult_64_64_128(op1, op2_bx);
            if (j == 0) {              
              accum_ax_ptr[i] = mul_ax;
              accum_bx_ptr[i] = mul_bx;
            }  else {
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
          
        //   for (uint64_t i = 0; i < param_degree * length; i++) {
        //     const int idx = i / param_degree;
        //     const int prime_idx =
        //         idx + ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
        //     const auto prime = primes_ptr[prime_idx];
        //     const auto barret_ratio = barret_ratio_ptr[prime_idx];
        //     const auto barret_k = barret_k_ptr[prime_idx];
        //     const auto res_ax = fhe::barret_reduction_128_64(
        //         accum_ax_ptr[i], prime, barret_ratio, barret_k);
        //     res_ax_ptr[i] = res_ax;
        //     const auto res_bx = fhe::barret_reduction_128_64(
        //         accum_bx_ptr[i], prime, barret_ratio, barret_k);
        //     res_bx_ptr[i] = res_bx;
        //   }
        }
      }),
      kUInt64);
}
Tensor innerproduct_cpu(
    const Tensor& res,
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
    const Tensor& workspace) {
  Tensor out = at::empty_like(res);
  out.resize_({2, (curr_limbs + alpha) * param_degree});
  innerproduct_template(
      modup_out,
      bx,
      ax,
      curr_limbs,
      alpha,
      level,
      param_degree,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      out);
  return out;
}

Tensor& innerproduct_cpu_(
    Tensor& res,
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
    const Tensor& workspace) {
  innerproduct_template(
      modup_out,
      bx,
      ax,
      curr_limbs,
      alpha,
      level,
      param_degree,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      res);
  return res;
}

Tensor& innerproduct_cpu_out(
    const Tensor& res,
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
    Tensor& out) {
  innerproduct_template(
      modup_out,
      bx,
      ax,
      curr_limbs,
      alpha,
      level,
      param_degree,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      out);
  return out;
}

} // namespace at::native