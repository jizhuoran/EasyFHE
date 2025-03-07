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
  cudaFHE::mod_raise_template(
      out.mutable_data_ptr<uint64_t>(),
      in.mutable_data_ptr<uint64_t>(),
      moduliQ.data_ptr<uint64_t>(),
      N,
      L0,
      logN,
      level,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>());
  return out;
}

} // namespace at::native