#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>

#include <cudaFHE.h>

namespace at::native {

Tensor innerproduct_cuda(
    const Tensor& res,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  Tensor out = at::empty_like(res);
  out.resize_({2, (curr_limbs + (primes.numel() - L)) * N});

  auto out_bx_ptr = reinterpret_cast<uint64_t*>(out[0].data_ptr<uint64_t>());
  auto out_ax_ptr = reinterpret_cast<uint64_t*>(out[1].data_ptr<uint64_t>());

  cudaFHE::innerproduct_template(
      in.data_ptr<uint64_t>(),
      bx.data_ptr<uint64_t>(),
      ax.data_ptr<uint64_t>(),
      curr_limbs,
      L,
      N,
      primes.numel(),
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      out_bx_ptr,
      out_ax_ptr
    );
  return out;
}

} // namespace at::native