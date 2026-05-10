#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorCompare.h>

namespace at::native {

namespace {

// Composite op implementation for simplicity. This materializes the cross product of elements and test elements,
// so it is not very memory efficient, but it is fast on CUDA.
void isin_default_kernel_gpu(
    const Tensor& elements, const Tensor& test_elements, bool invert, const Tensor& out) {
  TORCH_CHECK(false, "isin is disabled in EasyFHE");
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(isin_default_stub, &isin_default_kernel_gpu)

} // namespace at::native
