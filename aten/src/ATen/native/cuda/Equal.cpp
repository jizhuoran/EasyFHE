#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>

namespace at::native {

bool cuda_equal(const Tensor& self, const Tensor &src) {
  TORCH_CHECK(false, "equal is disabled in EasyFHE");
  return false;
}

} // namespace at::native
