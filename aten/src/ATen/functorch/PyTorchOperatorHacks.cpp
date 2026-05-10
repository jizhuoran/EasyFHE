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

}

TORCH_LIBRARY_IMPL(aten, FuncTorchDynamicLayerFrontMode, m) {
  m.impl("index_select_backward", index_select_backward_hack);
}

} // namespace at::functorch
