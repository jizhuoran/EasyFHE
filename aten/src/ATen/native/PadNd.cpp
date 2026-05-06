#include <ATen/native/Padding.h>

namespace at::native {

DEFINE_DISPATCH(reflection_pad1d_kernel);
DEFINE_DISPATCH(reflection_pad1d_backward_kernel);
DEFINE_DISPATCH(reflection_pad2d_kernel);
DEFINE_DISPATCH(reflection_pad2d_backward_kernel);
DEFINE_DISPATCH(reflection_pad3d_kernel);
DEFINE_DISPATCH(reflection_pad3d_backward_kernel);

} // namespace at::native
