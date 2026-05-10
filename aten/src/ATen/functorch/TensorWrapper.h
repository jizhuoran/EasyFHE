#pragma once

#include <ATen/Tensor.h>
#include <c10/macros/Export.h>
#include <stdexcept>

namespace at::functorch {

struct TORCH_API TensorWrapper {
  const Tensor& value() const {
    throw std::runtime_error("functorch TensorWrapper is disabled in EasyFHE");
  }
};

TORCH_API TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor);

} // namespace at::functorch
