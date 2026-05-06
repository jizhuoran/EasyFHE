#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexing.h>

namespace at::native {

DEFINE_DISPATCH(index_put_kernel_quantized_stub);
DEFINE_DISPATCH(masked_fill_kernel_quantized_stub);
DEFINE_DISPATCH(index_put_with_sort_quantized_stub);

} // namespace at::native
