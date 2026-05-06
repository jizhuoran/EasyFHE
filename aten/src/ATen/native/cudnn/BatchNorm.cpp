#include <ATen/core/Tensor.h>
#include <ATen/native/cudnn/BatchNorm.h>

namespace at::native {

size_t _get_cudnn_batch_norm_reserve_space_size(const Tensor& input_t, bool training) {
  return 0;
}

} // namespace at::native
