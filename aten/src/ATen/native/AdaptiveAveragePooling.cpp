#include <ATen/native/AdaptivePooling.h>

namespace at::native {

DEFINE_DISPATCH(adaptive_avg_pool2d_kernel);
DEFINE_DISPATCH(adaptive_avg_pool2d_backward_kernel);

} // namespace at::native
