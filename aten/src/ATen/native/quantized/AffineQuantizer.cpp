#include <ATen/native/quantized/AffineQuantizer.h>

namespace at::native {

DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub);

REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_tensor_affine_stub);
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_channel_affine_stub);
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_channel_float_qparams_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_tensor_affine_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_channel_affine_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub);
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub);

Tensor& quantize_tensor_per_tensor_affine(
    const Tensor& rtensor, Tensor& qtensor, double scale, int64_t zero_point) {
  TORCH_CHECK(false, "quantize_tensor_per_tensor_affine not supported in EasyFHE");
}

Tensor& quantize_tensor_per_channel_affine(
    const Tensor& rtensor, Tensor& qtensor, const Tensor& scales,
    Tensor zero_points, int64_t axis) {
  TORCH_CHECK(false, "quantize_tensor_per_channel_affine not supported in EasyFHE");
}

Tensor& quantize_tensor_per_channel_float_qparams(
    const Tensor& rtensor, Tensor& qtensor, const Tensor& scales,
    const Tensor& zero_points, int64_t axis) {
  TORCH_CHECK(false, "quantize_tensor_per_channel_float_qparams not supported in EasyFHE");
}

Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor& qtensor, Tensor& rtensor, double scale, int64_t zero_point) {
  TORCH_CHECK(false, "dequantize_tensor_per_tensor_affine not supported in EasyFHE");
}

Tensor& dequantize_tensor_per_channel_affine(
    const Tensor& qtensor, Tensor& rtensor, const Tensor& scales,
    Tensor zero_points, int64_t axis) {
  TORCH_CHECK(false, "dequantize_tensor_per_channel_affine not supported in EasyFHE");
}

Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor& qtensor, Tensor& rtensor, const Tensor& scales,
    const Tensor& zero_points, int64_t axis) {
  TORCH_CHECK(false, "dequantize_tensor_per_channel_float_qparams not supported in EasyFHE");
}

} // namespace at::native
