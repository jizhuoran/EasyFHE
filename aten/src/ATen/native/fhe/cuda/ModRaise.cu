#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cuda/CommonOperation.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

static void mod_raise_template(
    Tensor& res,
    const Tensor& in,
    const Tensor& moduliQ,
    int64_t N,
    int64_t L0,
    int64_t logN,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  auto op_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      res_ptr,
      0,
      1,
      1,
      level,
      N,
      inverse_power_of_roots_div_two,
      moduliQ,
      inverse_scaled_power_of_roots_div_two);

  switch_modulus(res_ptr, res_ptr, moduliQ, barret_ratio, barret_k, 0, L0, N);

  NTT_impl(
      res_ptr,
      res_ptr,
      L0,
      N,
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      moduliQ.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());
}

Tensor mod_raise_cuda(
    const Tensor& res,
    const Tensor& in,
    const Tensor& moduliQ,
    int64_t N,
    int64_t L0,
    int64_t logN,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  Tensor out = at::empty_like(res).resize_({L0 * N});
  //   out.resize_({2, (curr_limbs + alpha) * param_degree});
  mod_raise_template(
      out,
      in,
      moduliQ,
      N,
      L0,
      logN,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots,
      barret_ratio,
      barret_k);
  return out;
}

} // namespace at::native