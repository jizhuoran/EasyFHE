#include <ATen/functorch/TensorWrapper.h>

namespace at::functorch {

TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor) {
  return nullptr;
}

} // namespace at::functorch
