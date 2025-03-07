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

Tensor moddown_cuda(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    int64_t N,
    int64_t log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  auto out = at::empty(curr_limbs * N, in.options());
  auto workspace = at::empty((curr_limbs + (primes.numel() - L)) * N, in.options());
  cudaFHE::moddown_cuda_template(
      out.mutable_data_ptr<uint64_t>(),
      workspace.mutable_data_ptr<uint64_t>(),
      in.mutable_data_ptr<uint64_t>(),
      curr_limbs,
      L,
      N,
      log_degree,
      hat_inverse_vec_moddown.data_ptr<uint64_t>(),
      hat_inverse_vec_shoup_moddown.data_ptr<uint64_t>(),
      prod_q_i_mod_q_j_moddown.data_ptr<uint64_t>(),
      prod_inv_moddown.data_ptr<uint64_t>(),
      prod_inv_shoup_moddown.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>(),
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());
  return out;
}

} // namespace at::native