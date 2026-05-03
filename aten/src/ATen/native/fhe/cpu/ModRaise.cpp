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

static void mod_raise_template(
    Tensor& res,
    const Tensor& in,
    int64_t L0,
    uint64_t old_primes,
    const Tensor& moduliQ,
    const Tensor& switch_modulus_map,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  const auto num_cv = in.sizes()[0];
  const auto num_cipher = in.sizes()[1];
  const auto L_IN = in.sizes()[2];
  const auto N = in.sizes()[3];

  auto* in_ptr = in.data_ptr<uint64_t>();
  auto* res_ptr = res.data_ptr<uint64_t>();

  iNTT_impl(
      res_ptr,
      in_ptr,
      1,
      N,
      L0,
      L_IN,
      num_cv,
      num_cipher,
      moduliQ.data_ptr<uint64_t>(),
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());

  switch_modulus(
      res_ptr,
      res_ptr,
      0,
      L0,
      N,
      L0,
      L0,
      num_cv,
      num_cipher,
      old_primes >> 1,
      moduliQ,
      switch_modulus_map);

  NTT_impl(
      res_ptr,
      L0,
      N,
      L0,
      num_cv,
      num_cipher,
      moduliQ.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());
}

Tensor mod_raise_cpu(
    const Tensor& in,
    int64_t N,
    int64_t L0,
    int64_t old_primes,
    const Tensor& moduliQ,
    const Tensor& switch_modulus_map,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);

  auto out = at::empty({in.sizes()[0], in.sizes()[1], L0, N}, in.options());

  mod_raise_template(
      out,
      in,
      L0,
      static_cast<uint64_t>(old_primes),
      moduliQ,
      switch_modulus_map,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);

  return out;
}

} // namespace at::native
