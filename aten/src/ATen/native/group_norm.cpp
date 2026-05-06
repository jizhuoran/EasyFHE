#include <ATen/native/group_norm.h>

namespace at::native {

DEFINE_DISPATCH(GroupNormKernel);
DEFINE_DISPATCH(GroupNormBackwardKernel);

} // namespace at::native
