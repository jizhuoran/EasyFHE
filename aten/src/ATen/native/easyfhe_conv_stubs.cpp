#include <ATen/native/ConvUtils.h>
#include <ATen/native/Normalization.h>

namespace at::native {

ConvBackend select_conv_backend(
    const Tensor&,
    const Tensor&,
    const std::optional<Tensor>&,
    SymIntArrayRef,
    SymIntArrayRef,
    SymIntArrayRef,
    bool,
    SymIntArrayRef,
    c10::SymInt,
    const at::OptionalSymIntArrayRef) {
  return ConvBackend::Empty;
}

at::MemoryFormat _determine_backend_memory_format(
    const Tensor&,
    const Tensor&,
    const ConvBackend) {
  return at::MemoryFormat::Contiguous;
}

void _cudnn_set_conv_benchmark_empty_cache(bool) {}

bool _cudnn_get_conv_benchmark_empty_cache() {
  return false;
}

BatchNormBackend _select_batch_norm_backend(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    bool,
    double) {
  return BatchNormBackend::Native;
}

TORCH_API size_t _get_cudnn_batch_norm_reserve_space_size(const Tensor&, bool) {
  return 0;
}

} // namespace at::native
