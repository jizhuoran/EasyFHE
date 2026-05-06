#include <ATen/native/AmpKernels.h>

namespace at::native {

DEFINE_DISPATCH(_amp_foreach_non_finite_check_and_unscale_cpu_stub);
DEFINE_DISPATCH(_amp_update_scale_cpu_stub);

} // namespace at::native
