#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <c10/util/Exception.h>

namespace at::native::detail {

int64_t cufft_get_plan_cache_max_size_impl(DeviceIndex device_index) {
  return 0;
}

void cufft_set_plan_cache_max_size_impl(DeviceIndex device_index, int64_t max_size) {
}

int64_t cufft_get_plan_cache_size_impl(DeviceIndex device_index) {
  return 0;
}

void cufft_clear_plan_cache_impl(DeviceIndex device_index) {
}

} // namespace at::native::detail
