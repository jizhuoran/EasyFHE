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
  
Tensor modup_cuda(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t N,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  auto out = at::empty((curr_limbs + primes.numel() - L) * N, in.options());
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());

  cudaFHE::modup_cuda_template(
      out_ptr,
      in_ptr,
      curr_limbs,
      L,
      hat_inverse_vecs.data_ptr<uint64_t>(),
      hat_inverse_vecs.strides()[0],
      hat_inverse_vec_shoups.data_ptr<uint64_t>(),
      hat_inverse_vec_shoups.strides()[0],
      prod_q_i_mod_q_js.data_ptr<uint64_t>(),
      prod_q_i_mod_q_js.strides()[0],
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      N,
      primes.numel(),
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());
  return out;
}

} // namespace at::native