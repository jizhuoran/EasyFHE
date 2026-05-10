#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/quantized/QTensorImpl.h>

namespace at {

namespace {

[[noreturn]] void unsupported_quantized() {
  TORCH_CHECK(false, "EasyFHE does not support quantized tensors");
}

} // namespace

QTensorImpl::QTensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), std::move(key_set), data_type),
      quantizer_(std::move(quantizer)) {}

QTensorImpl::QTensorImpl(
    ImplType type,
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    QuantizerPtr quantizer)
    : TensorImpl(type, std::move(storage), std::move(key_set), data_type),
      quantizer_(std::move(quantizer)) {}

const char* QTensorImpl::tensorimpl_type_name() const {
  return "QTensorImpl";
}

QuantizerPtr TensorBase::quantizer() const {
  unsupported_quantized();
}

QuantizerPtr make_per_tensor_affine_quantizer(
    double,
    int64_t,
    ScalarType) {
  unsupported_quantized();
}

QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor&,
    const Tensor&,
    int64_t,
    ScalarType) {
  unsupported_quantized();
}

QuantizerPtr make_unknown_quantizer(ScalarType) {
  unsupported_quantized();
}

QTensorImpl* get_qtensorimpl(const TensorBase&) {
  unsupported_quantized();
}

Tensor new_qtensor(IntArrayRef, const TensorOptions&, QuantizerPtr) {
  unsupported_quantized();
}

void set_quantizer_(const Tensor&, ConstQuantizerPtr) {
  unsupported_quantized();
}

Tensor from_blob_quantized_per_tensor_affine(
    void*,
    IntArrayRef,
    IntArrayRef,
    std::function<void(void*)>,
    const float,
    const int64_t,
    const TensorOptions&) {
  unsupported_quantized();
}

Tensor from_blob_quantized_per_tensor_affine(
    void*,
    IntArrayRef,
    std::function<void(void*)>,
    const float,
    const int64_t,
    const TensorOptions&) {
  unsupported_quantized();
}

Tensor from_blob_quantized_per_channel_affine(
    void*,
    IntArrayRef,
    std::function<void(void*)>,
    const Tensor&,
    const Tensor&,
    const int64_t,
    const TensorOptions&) {
  unsupported_quantized();
}

Tensor UnknownQuantizer::quantize(const Tensor&) {
  unsupported_quantized();
}

Tensor UnknownQuantizer::dequantize(const Tensor&) {
  unsupported_quantized();
}

Tensor& UnknownQuantizer::dequantize_out(Tensor&, const Tensor&) {
  unsupported_quantized();
}

QScheme UnknownQuantizer::qscheme() const {
  unsupported_quantized();
}

bool UnknownQuantizer::equalTo(QuantizerPtr) const {
  unsupported_quantized();
}

Tensor PerTensorAffineQuantizer::quantize(const Tensor&) {
  unsupported_quantized();
}

Tensor PerTensorAffineQuantizer::dequantize(const Tensor&) {
  unsupported_quantized();
}

Tensor& PerTensorAffineQuantizer::dequantize_out(Tensor&, const Tensor&) {
  unsupported_quantized();
}

Tensor PerChannelAffineQuantizer::quantize(const Tensor&) {
  unsupported_quantized();
}

Tensor PerChannelAffineQuantizer::dequantize(const Tensor&) {
  unsupported_quantized();
}

Tensor& PerChannelAffineQuantizer::dequantize_out(Tensor&, const Tensor&) {
  unsupported_quantized();
}

Tensor PerChannelAffineFloatQParamsQuantizer::quantize(const Tensor&) {
  unsupported_quantized();
}

Tensor PerChannelAffineFloatQParamsQuantizer::dequantize(const Tensor&) {
  unsupported_quantized();
}

Tensor& PerChannelAffineFloatQParamsQuantizer::dequantize_out(
    Tensor&,
    const Tensor&) {
  unsupported_quantized();
}

namespace native {

DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub);
DEFINE_DISPATCH(index_put_kernel_quantized_stub);
DEFINE_DISPATCH(masked_fill_kernel_quantized_stub);
DEFINE_DISPATCH(index_put_with_sort_quantized_stub);

REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_tensor_affine_stub);
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_channel_affine_stub);
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_channel_float_qparams_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_tensor_affine_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_channel_affine_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub);
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub);
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub);

Tensor& quantize_tensor_per_tensor_affine(
    const Tensor&,
    Tensor&,
    double,
    int64_t) {
  unsupported_quantized();
}

Tensor& quantize_tensor_per_channel_affine(
    const Tensor&,
    Tensor&,
    const Tensor&,
    Tensor,
    int64_t) {
  unsupported_quantized();
}

Tensor& quantize_tensor_per_channel_float_qparams(
    const Tensor&,
    Tensor&,
    const Tensor&,
    const Tensor&,
    int64_t) {
  unsupported_quantized();
}

Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor&,
    Tensor&,
    double,
    int64_t) {
  unsupported_quantized();
}

Tensor& dequantize_tensor_per_channel_affine(
    const Tensor&,
    Tensor&,
    const Tensor&,
    Tensor,
    int64_t) {
  unsupported_quantized();
}

Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor&,
    Tensor&,
    const Tensor&,
    const Tensor&,
    int64_t) {
  unsupported_quantized();
}

template <typename T>
T quantize_val(double, int64_t, float) {
  unsupported_quantized();
}

template <typename T>
T quantize_val_arm(const float, const int32_t, const float) {
  unsupported_quantized();
}

template <typename T, int precision>
void quantize_vec(double, int64_t, const float*, T*, size_t) {
  unsupported_quantized();
}

template <typename T>
float dequantize_val(double, int64_t, T) {
  unsupported_quantized();
}

template <typename SRC_T, typename DST_T>
DST_T requantize_val(double, int64_t, double, int64_t, SRC_T) {
  unsupported_quantized();
}

template <typename DST_T>
DST_T requantize_from_int(double, int64_t, int64_t) {
  unsupported_quantized();
}

int quantize_val_float_qparams(float, float, float, int, int) {
  unsupported_quantized();
}

template TORCH_API qint8
quantize_val<qint8>(double, int64_t, float);
template TORCH_API quint8
quantize_val<quint8>(double, int64_t, float);
template TORCH_API qint32
quantize_val<qint32>(double, int64_t, float);
template TORCH_API uint8_t
quantize_val_arm<uint8_t>(const float, const int32_t, const float);
template TORCH_API int8_t
quantize_val_arm<int8_t>(const float, const int32_t, const float);
template TORCH_API void quantize_vec<c10::qint8>(
    double,
    int64_t,
    const float*,
    c10::qint8*,
    size_t);
template TORCH_API void quantize_vec<c10::quint8>(
    double,
    int64_t,
    const float*,
    c10::quint8*,
    size_t);
template TORCH_API void quantize_vec<c10::qint32, 32>(
    double,
    int64_t,
    const float*,
    c10::qint32*,
    size_t);
template TORCH_API float dequantize_val<qint8>(double, int64_t, qint8);
template TORCH_API float dequantize_val<quint8>(double, int64_t, quint8);
template TORCH_API float dequantize_val<qint32>(double, int64_t, qint32);
template TORCH_API qint8
requantize_val<qint8, qint8>(double, int64_t, double, int64_t, qint8);
template TORCH_API quint8
requantize_val<qint8, quint8>(double, int64_t, double, int64_t, qint8);
template TORCH_API qint32
requantize_val<qint8, qint32>(double, int64_t, double, int64_t, qint8);
template TORCH_API qint8
requantize_val<quint8, qint8>(double, int64_t, double, int64_t, quint8);
template TORCH_API quint8
requantize_val<quint8, quint8>(double, int64_t, double, int64_t, quint8);
template TORCH_API qint32
requantize_val<quint8, qint32>(double, int64_t, double, int64_t, quint8);
template TORCH_API qint8
requantize_val<qint32, qint8>(double, int64_t, double, int64_t, qint32);
template TORCH_API quint8
requantize_val<qint32, quint8>(double, int64_t, double, int64_t, qint32);
template TORCH_API qint32
requantize_val<qint32, qint32>(double, int64_t, double, int64_t, qint32);
template TORCH_API qint8 requantize_from_int<qint8>(double, int64_t, int64_t);
template TORCH_API quint8
requantize_from_int<quint8>(double, int64_t, int64_t);
template TORCH_API qint32 requantize_from_int<qint32>(double, int64_t, int64_t);

} // namespace native
} // namespace at
