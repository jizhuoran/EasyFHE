#include <torch/csrc/utils/nested.h>
#include <ATen/ATen.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::utils {

at::Tensor nested_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    torch::PythonArgs& r) {
  TORCH_CHECK(false, "nested_tensor is not supported in this build");
}

} // namespace torch::utils
