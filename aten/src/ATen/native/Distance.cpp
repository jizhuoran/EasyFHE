#include <ATen/native/Distance.h>

namespace at::native {

DEFINE_DISPATCH(cdist_stub);
DEFINE_DISPATCH(cdist_backward_stub);
DEFINE_DISPATCH(pdist_forward_stub);
DEFINE_DISPATCH(pdist_backward_stub);

} // namespace at::native
