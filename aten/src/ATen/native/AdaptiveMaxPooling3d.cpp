#include <ATen/native/AdaptivePooling.h>

namespace at::native {

DEFINE_DISPATCH(adaptive_max_pool3d_kernel);
DEFINE_DISPATCH(adaptive_max_pool3d_backward_kernel);

} // namespace at::native
