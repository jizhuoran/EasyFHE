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

namespace at::native {

Tensor mul_by_monomial_cuda(
    const Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  
  TORCH_INTERNAL_ASSERT(false, "mul_by_monomial_cuda only supports inplace operation");
  return res;
}

Tensor& mul_by_monomial_cuda_(
    Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  Tensor temp = at::empty_like(res);
  cudaFHE::mul_by_monomial_inplace_template(
      res.mutable_data_ptr<uint64_t>(),
      temp.mutable_data_ptr<uint64_t>(),
      param_primes.data_ptr<uint64_t>(),
      l,
      N,
      M,
      monomialDeg,
      level,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());
  return res;
}

Tensor& mul_by_monomial_cuda_out(
    const Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    Tensor& out) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented");
  return out;
}

} // namespace at::native