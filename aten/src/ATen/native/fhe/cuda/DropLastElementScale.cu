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
#include <cudaFHE.h>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

Tensor drop_last_element_scale_cuda(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  auto res = at::empty((curr_limbs - 1) * N, to.options());
  auto workspace = at::empty(curr_limbs * N, to.options());

  cudaFHE::drop_last_element_scale_template(
      from.mutable_data_ptr<uint64_t>(),
      workspace.mutable_data_ptr<uint64_t>(),
      curr_limbs,
      l,
      L,
      N,
      param_primes.data_ptr<uint64_t>(),
      param_barret_ratio.data_ptr<uint64_t>(),
      param_barret_k.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>(),
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>(),
      qlql_inv_mod_ql_div_ql_mod_q.data_ptr<uint64_t>(),
      qlql_inv_mod_ql_div_ql_mod_q_shoup.data_ptr<uint64_t>(),
      q_inv_mod_q.data_ptr<uint64_t>(),
      q_inv_mod_q_shoup.data_ptr<uint64_t>(),
      res.mutable_data_ptr<uint64_t>());

  return res;
}
} // namespace at::native