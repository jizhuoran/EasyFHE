#pragma once

#include <ATen/Dispatch.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

namespace at::native {

TORCH_API Tensor& quantize_tensor_per_tensor_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point);
TORCH_API Tensor& quantize_tensor_per_channel_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis);
TORCH_API Tensor& quantize_tensor_per_channel_float_qparams(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);
TORCH_API Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point);
TORCH_API Tensor& dequantize_tensor_per_channel_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis);
TORCH_API Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using quantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor&, Tensor&, double, int64_t);
using quantize_tensor_per_channel_affine_fn =
    void (*)(const Tensor&, Tensor&, const Tensor&, const Tensor&, int64_t);
using quantize_tensor_per_channel_float_qparams_fn =
    void (*)(const Tensor&, Tensor&, const Tensor&, const Tensor&, int64_t);
using dequantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor&, Tensor&, double, int64_t);
using dequantize_tensor_per_channel_affine_fn =
    void (*)(const Tensor&, Tensor&, const Tensor&, const Tensor&, int64_t);
using dequantize_tensor_per_channel_float_qparams_fn =
    void (*)(const Tensor&, Tensor&, const Tensor&, const Tensor&, int64_t);
using quantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor&, Tensor&, float, float);
using dequantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor&, Tensor&, float, float);

DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_fn,
    quantize_tensor_per_tensor_affine_stub)
DECLARE_DISPATCH(
    quantize_tensor_per_channel_affine_fn,
    quantize_tensor_per_channel_affine_stub)
DECLARE_DISPATCH(
    quantize_tensor_per_channel_float_qparams_fn,
    quantize_tensor_per_channel_float_qparams_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_fn,
    dequantize_tensor_per_tensor_affine_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_affine_fn,
    dequantize_tensor_per_channel_affine_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_fn,
    dequantize_tensor_per_channel_float_qparams_stub)
DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_fn,
    quantize_tensor_per_tensor_affine_sub_byte_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_fn,
    dequantize_tensor_per_tensor_affine_sub_byte_stub)

using masked_fill_kernel_quantized_fn =
    void (*)(TensorIterator&, const Scalar&, double, int);
using index_put_kernel_quantized_fn =
    void (*)(TensorIterator&, IntArrayRef, IntArrayRef, bool, double, int);

DECLARE_DISPATCH(masked_fill_kernel_quantized_fn, masked_fill_kernel_quantized_stub)
DECLARE_DISPATCH(index_put_kernel_quantized_fn, index_put_kernel_quantized_stub)

template <typename T>
TORCH_API T quantize_val(double scale, int64_t zero_point, float value);
template <typename T>
T quantize_val_arm(float scale, int32_t zero_point, float value);
template <typename T, int precision = 8>
void quantize_vec(double scale, int64_t zero_point, const float* src, T* dst, size_t count = 8);
template <typename T>
TORCH_API float dequantize_val(double scale, int64_t zero_point, T value);
template <typename T>
TORCH_API float dequantize_vec(double scale, int64_t zero_point, const T* src, float* dst, size_t count = 8);
template <typename SRC_T, typename DST_T>
TORCH_API DST_T requantize_val(double src_scale, int64_t src_zero_point, double dst_scale, int64_t dst_zero_point, SRC_T src);
template <typename DST_T>
TORCH_API DST_T requantize_from_int(double multiplier, int64_t zero_point, int64_t src);

int quantize_val_float_qparams(float scale, float zero_point, float value, int qmin, int qmax);

} // namespace at::native
