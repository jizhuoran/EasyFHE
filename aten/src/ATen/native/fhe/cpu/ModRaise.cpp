#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cpu/CommonOperation.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

static void switch_modulus_template(
    Tensor& res,
    const Tensor& in,
    int64_t N,
    int64_t L0,
    int64_t logN,
    int64_t level,
    const Tensor& moduliQ,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  auto op = in.clone();
  auto op_ptr = reinterpret_cast<uint64_t*>(op.data_ptr<uint64_t>());
  auto res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      op_ptr,
      0,
      1,
      1,
      level,
      N,
      inverse_power_of_roots_div_two,
      moduliQ,
      inverse_scaled_power_of_roots_div_two);

  switch_modulus(op_ptr, res_ptr, moduliQ, 0, L0, N);

  NTT_impl(
      res_ptr,
      res_ptr,
      0,
      L0,
      N,
      param_power_of_roots_shoup,
      moduliQ,
      param_power_of_roots);
}

Tensor mod_raise_cpu(
    const Tensor& res,
    const Tensor& in,
    int64_t N,
    int64_t L0,
    int64_t logN,
    int64_t level,
    const Tensor& moduliQ,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  Tensor out = at::empty_like(res).resize_({L0 * N});
  //   out.resize_({2, (curr_limbs + alpha) * param_degree});
  switch_modulus_template(
      out,
      in,
      N,
      L0,
      logN,
      level,
      moduliQ,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

Tensor& switch_modulus_cpu_(
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
    const Tensor& param_power_of_roots) {
  res.resize_({L0 * N});
  switch_modulus_template(
      res,
      in,
      N,
      L0,
      logN,
      level,
      moduliQ,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return res;
}

Tensor& switch_modulus_cpu_out(
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
    Tensor& out) {
  out.resize_({L0 * N});
  switch_modulus_template(
      out,
      in,
      N,
      L0,
      logN,
      level,
      moduliQ,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

} // namespace at::native
