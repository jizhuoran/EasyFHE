#include <ATen/native/Pool.h>

namespace at::native {

DEFINE_DISPATCH(max_pool2d_kernel);
DEFINE_DISPATCH(max_pool2d_backward_kernel);

} // namespace at::native
