#include <c10/macros/Export.h>

namespace at::native {

TORCH_API bool is_vulkan_available() {
  return false;
}

} // namespace at::native
