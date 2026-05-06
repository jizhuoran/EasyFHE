#include <ATen/native/Pool.h>

namespace at::native {

DEFINE_DISPATCH(avg_pool2d_kernel);
DEFINE_DISPATCH(avg_pool2d_backward_kernel);

} // namespace at::native
