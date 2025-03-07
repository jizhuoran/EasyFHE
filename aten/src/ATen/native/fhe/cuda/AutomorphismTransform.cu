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

Tensor automorphism_transform_cuda(
    const Tensor& a,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  Tensor out = at::empty_like(a);
  cudaFHE::automorphism_transform_template(out.data_ptr<uint64_t>(), a.data_ptr<uint64_t>(), l, N, precomp_vec.data_ptr<int32_t>());
  return out;
}

} // namespace at::native