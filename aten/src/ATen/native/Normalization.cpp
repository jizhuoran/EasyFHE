#include <ATen/native/Normalization.h>
#include <ATen/native/batch_norm.h>
#include <ATen/core/Tensor.h>

namespace at::native {

DEFINE_DISPATCH(renorm_scale_factor_stub);
DEFINE_DISPATCH(batch_norm_cpu_stub);
DEFINE_DISPATCH(batch_norm_cpu_collect_stats_stub);
DEFINE_DISPATCH(batch_norm_cpu_backward_stub);

BatchNormBackend _select_batch_norm_backend(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool training, double eps) {
  return BatchNormBackend::Native;
}

} // namespace at::native
