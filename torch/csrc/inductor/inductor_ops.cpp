#include <torch/csrc/inductor/inductor_ops.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorImpl.h>

namespace torch::inductor {
using namespace at;

Tensor _mm_plus_mm_out(
    Tensor& out,
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& d) {
  TORCH_CHECK(false, "_mm_plus_mm_out not supported in EasyFHE");
  return out;
}

Tensor _alloc_from_pool(
    const Tensor& self,
    int64_t offset_bytes,
    ScalarType dtype,
    IntArrayRef size,
    IntArrayRef stride) {
  TORCH_CHECK(self.storage_offset() == 0);
  Tensor self_ = at::detail::make_tensor<TensorImpl>(
      Storage(self.storage()),
      self.key_set(),
      caffe2::TypeMeta::fromScalarType(dtype));
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(
      offset_bytes / static_cast<int64_t>(c10::elementSize(dtype)));
  self_tmp_->set_sizes_and_strides(size, stride);
  return self_;
}

Tensor _reinterpret_tensor(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    int64_t offset_increment) {
  Tensor self_ = at::detail::make_tensor<TensorImpl>(
      Storage(self.storage()), self.key_set(), self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset() + offset_increment);
  self_tmp_->set_sizes_and_strides(size, stride);
  return self_;
}

} // namespace torch::inductor
