#include <ATen/native/Pool.h>

namespace at::native {

DEFINE_DISPATCH(max_pool3d_kernel);
DEFINE_DISPATCH(max_pool3d_backward_kernel);

} // namespace at::native
