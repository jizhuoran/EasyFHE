#include <ATen/functorch/DynamicLayer.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/Dispatch.h>
#include <c10/util/irange.h>
#include <c10/util/Exception.h>
#include <ATen/NamedTensorUtils.h>

namespace at::functorch {

namespace {
Tensor index_select_backward_hack(const Tensor& grad, IntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  return at::zeros(self_sizes, grad.options()).index_add(dim, index, grad);
}

Tensor trace_backward_decomp(const Tensor& grad, IntArrayRef sizes) {
  TORCH_CHECK(sizes.size() == 2, "expected matrix input");
  auto grad_input = at::zeros(sizes[0] * sizes[1], grad.options());
  auto diag_size = std::min(sizes[0], sizes[1]);
  auto step = sizes[1] + 1;
  auto indices = at::arange(0, diag_size * step, step, grad.options().dtype(at::kLong));
  grad_input = grad_input.index_put({indices}, grad);
  return grad_input.view(sizes);
}
}

TORCH_LIBRARY_IMPL(aten, FuncTorchDynamicLayerFrontMode, m) {
  m.impl("index_select_backward", index_select_backward_hack);
  m.impl("trace_backward", trace_backward_decomp);
}

} // namespace at::functorch
