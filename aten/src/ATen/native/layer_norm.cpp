#include <ATen/native/layer_norm.h>

namespace at::native {

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

} // namespace at::native
