#include <ATen/native/ConvUtils.h>
#include <ATen/core/Tensor.h>

namespace at::native {

DEFINE_DISPATCH(conv_depthwise2d_backward_stub);
DEFINE_DISPATCH(conv_depthwise3d_backward_stub);
DEFINE_DISPATCH(cudnn_convolution_backward_stub);
DEFINE_DISPATCH(cudnn_convolution_transpose_backward_stub);
DEFINE_DISPATCH(miopen_convolution_backward_stub);
DEFINE_DISPATCH(miopen_convolution_transpose_backward_stub);
DEFINE_DISPATCH(miopen_depthwise_convolution_backward_stub);
DEFINE_DISPATCH(mkldnn_convolution_backward_stub);
DEFINE_DISPATCH(mkldnn_convolution_transpose_backward_stub);
DEFINE_DISPATCH(mkldnn_convolution_transpose_stub);
DEFINE_DISPATCH(mps_convolution_backward_stub);
DEFINE_DISPATCH(slow_conv_dilated2d_backward_stub);
DEFINE_DISPATCH(slow_conv_dilated3d_backward_stub);
DEFINE_DISPATCH(slow_conv_transpose2d_backward_stub);
DEFINE_DISPATCH(slow_conv_transpose3d_backward_stub);

at::MemoryFormat _determine_backend_memory_format(
    const Tensor& input,
    const Tensor& weight,
    const ConvBackend backend) {
  return at::MemoryFormat::Contiguous;
}

ConvBackend select_conv_backend(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation,
    bool transposed, SymIntArrayRef output_padding, c10::SymInt groups,
    const at::OptionalSymIntArrayRef bias_sizes_opt) {
  return ConvBackend::Slow2d;
}

static bool conv_benchmark_empty_cache = true;

void _cudnn_set_conv_benchmark_empty_cache(bool enable) {
  conv_benchmark_empty_cache = enable;
}

bool _cudnn_get_conv_benchmark_empty_cache() {
  return conv_benchmark_empty_cache;
}

} // namespace at::native
