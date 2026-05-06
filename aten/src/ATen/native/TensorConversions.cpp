// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_autocast_to_full_precision_native.h>
#include <ATen/ops/_autocast_to_reduced_precision_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/_to_cpu_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/core/ATen_fwd.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/TensorConversions.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <algorithm>
#include <numeric>

namespace at::native {

static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline std::optional<Device> ensure_has_index(
    std::optional<Device> device) {
  if (!device.has_value()) {
    return std::nullopt;
  }
  return ensure_has_index(device.value());
}

Tensor _to_copy(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !layout.has_value() || self.layout() == layout.value(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      layout.value());
  auto options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  // memory_format is handled separately due to MemoryFormat::Preserve logic
  options = self.options().merge_in(options).memory_format(std::nullopt);
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  // TODO: Use the dispatcher for this.
  // Currently there are unenumerated extensibility issues preventing this.
  if (self.layout() != kStrided) {
    TORCH_CHECK(false, "Non-strided tensors not supported in EasyFHE");
  }

  bool pin_out =
      (non_blocking &&
       at::accelerator::isAcceleratorExcluded(self.device().type(), at::kMPS) &&
       options.device().is_cpu() && (options.layout() == c10::kStrided));

  if (memory_format == MemoryFormat::Preserve) {
    if (options.device().supports_as_strided()) {
      if (self.is_non_overlapping_and_dense()) {
        Tensor r;
        r = at::empty_strided(
            self.sizes(), self.strides(), options.pinned_memory(pin_out));
        r.copy_(self, non_blocking);
        return r;
      } else if (self.layout() == kStrided) {
        Tensor r;
        auto strides = infer_dense_strides(self.sizes(), self.strides());
        r = at::empty_strided(
            self.sizes(), strides, options.pinned_memory(pin_out));
        r.copy_(self, non_blocking);
        return r;
      } else {
        memory_format = self.suggest_memory_format();
      }
    } else {
      memory_format = self.suggest_memory_format();
    }
  }
  auto r = at::empty_symint(
            self.sym_sizes(),
            options.memory_format(memory_format).pinned_memory(pin_out),
            std::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

template <typename T>
static inline bool is_null_or_equal_to(
    const std::optional<T>& test,
    const T& value) {
  if (!test.has_value()) {
    return true;
  }
  return test.value() == value;
}

// NOTE: static runtime's to_maybe_copy_out relies on details of this
// check; if you change how it works, please update static runtime as
// well.
bool to_will_alias(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  return is_null_or_equal_to(dtype, self.dtype().toScalarType()) &&
      is_null_or_equal_to(layout, self.layout()) &&
      is_null_or_equal_to(device, self.device()) && !copy &&
      (memory_format == MemoryFormat::Preserve ||
       self.suggest_memory_format() == memory_format);
}

static inline Tensor to_impl(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // fast path
  if (to_will_alias(
          self, dtype, layout, device, copy, optional_memory_format)) {
    return self;
  }
  return at::_to_copy(
      self,
      dtype,
      layout,
      device,
      pin_memory,
      non_blocking,
      optional_memory_format);
}

// If input tensor is fp32, cast it to fp16, otherwise leave it alone.
// (this is intended to be used internally by the JIT autocast implementation)
Tensor _autocast_to_reduced_precision(
    const Tensor& self,
    bool cuda_enabled,
    bool cpu_enabled,
    ScalarType cuda_dtype,
    ScalarType cpu_dtype) {
  if (self.dtype() == at::ScalarType::Float &&
      ((self.device().is_cuda() && cuda_enabled) ||
       (self.device().is_cpu() && cpu_enabled))) {
    at::ScalarType target = at::ScalarType::Undefined;
    if (self.device().is_cuda()) {
      target = cuda_dtype;
    } else if (self.device().is_cpu()) {
      target = cpu_dtype;
    }

    TORCH_INTERNAL_ASSERT(
        target != at::ScalarType::Undefined,
        "_autocast_to_reduced_precision requires legit ScalarType argument for given device");

    return to_impl(
        self,
        target,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        false,
        false,
        std::nullopt);
  } else {
    return self;
  }
}

// If input tensor is fp16, cast it to fp32, otherwise leave it alone.
// (this is intended to be used internally by the JIT autocast implementation)
Tensor _autocast_to_full_precision(
    const Tensor& self,
    bool cuda_enabled,
    bool cpu_enabled) {
  if ((self.dtype() == at::ScalarType::Half ||
       self.dtype() == at::ScalarType::BFloat16) &&
      ((self.device().is_cuda() && cuda_enabled) ||
       (self.device().is_cpu() && cpu_enabled))) {
    return to_impl(
        self,
        at::ScalarType::Float,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        false,
        false,
        std::nullopt);
  } else {
    return self;
  }
}

Tensor to(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      layout,
      ensure_has_index(device),
      pin_memory,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(
    const Tensor& self,
    Device device,
    ScalarType dtype,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      std::nullopt,
      ensure_has_index(device),
      std::nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(
    const Tensor& self,
    ScalarType dtype,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(
    const Tensor& self,
    const Tensor& other,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = other.options();
  return to_impl(
      self,
      options.dtype().toScalarType(),
      options.layout(),
      options.device(),
      options.pinned_memory(),
      non_blocking,
      copy,
      optional_memory_format);
}

// This op is important primarily for lazy / graph-based backends.
// While this vanilla implementation loops through each tensor and independently
// converts it to cpu, a lazy backend like XLA might need to tell sync updates
// across tensors.
std::vector<Tensor> _to_cpu(TensorList tensors) {
  std::vector<Tensor> cpu_tensors;
  for (const auto& t : tensors) {
    cpu_tensors.push_back(t.cpu());
  }
  return cpu_tensors;
}

Tensor to_dense(
    const Tensor& tensor,
    std::optional<c10::ScalarType> dtype,
    std::optional<bool> masked_grad) {
  TORCH_CHECK(
      tensor.layout() == c10::kStrided,
      "to_dense does not support layout ",
      tensor.layout());
  if (dtype) {
    return tensor.to(*dtype);
  }
  return tensor;
}

// Computes the strides for view_dtype output when the view dtype is
// smaller than the original dtype
static inline SymDimVector compute_strides_for_view_dtype_downsize(
    SymIntArrayRef old_strides,
    int64_t size_ratio,
    ScalarType old_dtype,
    ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();

  TORCH_CHECK(
      old_strides[ndim - 1] == 1,
      "self.stride(-1) must be 1 to view ",
      old_dtype,
      " as ",
      new_dtype,
      " (different element sizes), but got ",
      old_strides[ndim - 1]);

  SymDimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    new_strides[dim_idx] = old_strides[dim_idx] * size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

// Computes the strides for view_dtype output when the view dtype is
// larger than the original dtype
static inline SymDimVector compute_strides_for_view_dtype_upsize(
    SymIntArrayRef old_strides,
    int64_t size_ratio,
    ScalarType old_dtype,
    ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();
  TORCH_CHECK(
      old_strides[ndim - 1] == 1,
      "self.stride(-1) must be 1 to view ",
      old_dtype,
      " as ",
      new_dtype,
      " (different element sizes), but got ",
      old_strides[ndim - 1]);

  SymDimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    TORCH_CHECK(
        (old_strides[dim_idx] % size_ratio) == 0,
        "self.stride(",
        dim_idx,
        ") must be divisible by ",
        size_ratio,
        " to view ",
        old_dtype,
        " as ",
        new_dtype,
        " (different element sizes), ",
        "but got ",
        old_strides[dim_idx]);

    new_strides[dim_idx] = old_strides[dim_idx] / size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

Tensor view_dtype(const Tensor& self, ScalarType dtype) {
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  TORCH_CHECK(
      !self.is_conj(),
      "torch.Tensor.view is not supported for conjugate view tensors when converting to a different dtype.");
  TORCH_CHECK(
      !self.is_neg(),
      "torch.Tensor.view is not supported for tensors with negative bit set when converting to a different dtype.");

  int64_t self_element_size = self.element_size();
  int64_t new_element_size = static_cast<int64_t>(type_meta.itemsize());

  Storage storage = self.storage();
  auto new_tensor = detail::make_tensor<TensorImpl>(
      std::move(storage), self.key_set(), type_meta);
  auto* impl = new_tensor.unsafeGetTensorImpl();

  if (self_element_size == new_element_size) {
    impl->set_sizes_and_strides(
        self.sym_sizes(), self.sym_strides(), self.sym_storage_offset());

  } else if (self.dim() == 0) {
    TORCH_CHECK(
        false,
        "self.dim() cannot be 0 to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes)");

  } else if (self_element_size > new_element_size) {
    // Downsizing element size

    int64_t size_ratio = self_element_size / new_element_size;
    auto new_strides = compute_strides_for_view_dtype_downsize(
        self.sym_strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sym_sizes();
    SymDimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] *= size_ratio;

    auto new_storage_offset = size_ratio * self.sym_storage_offset();

    impl->set_sizes_and_strides(new_sizes, new_strides, new_storage_offset);

  } else {
    // Upsizing element size

    int64_t size_ratio = new_element_size / self_element_size;

    TORCH_CHECK(
        (self.sym_size(-1) % size_ratio) == 0,
        "self.size(-1) must be divisible by ",
        size_ratio,
        " to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes), ",
        "but got ",
        self.sym_size(-1));

    TORCH_CHECK(
        (self.sym_storage_offset() % size_ratio) == 0,
        "self.storage_offset() must be divisible by ",
        size_ratio,
        " to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes), but got ",
        self.sym_storage_offset());

    auto new_strides = compute_strides_for_view_dtype_upsize(
        self.sym_strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sym_sizes();
    SymDimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] /= size_ratio;

    auto new_storage_offset = self.sym_storage_offset() / size_ratio;

    impl->set_sizes_and_strides(new_sizes, new_strides, new_storage_offset);
  }

  return new_tensor;
}


Tensor to_meta(const Tensor& tensor) {
  auto out = at::native::empty_strided_meta_symint(
      tensor.sym_sizes(),
      tensor.sym_strides(),
      /*dtype=*/tensor.scalar_type(),
      /*layout=*/tensor.layout(),
      /*device=*/c10::Device(c10::kMeta),
      /*pin_memory=*/std::nullopt);
  // needs to handle wrapped numbers, so dtype promotion works properly.
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}
std::optional<Tensor> to_meta(const std::optional<Tensor>& tensor) {
  if (tensor.has_value()) {
    return to_meta(*tensor);
  }
  return std::nullopt;
}

std::vector<Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<Tensor> outs;
  outs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outs.push_back(to_meta(tensor));
  }
  return outs;
}
} // namespace at::native
