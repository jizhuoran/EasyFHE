#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/accumulate.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/Loops.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_efficientzerotensor_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/eye_native.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace at::native {

Tensor& eye_out_cuda(int64_t n, Tensor& result) {
  // the default value of `m` equals to `n`
  return at::native::eye_out_cuda(n, n, result);
}

Tensor& eye_out_cuda(int64_t n, int64_t m, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

Tensor empty_cuda(IntArrayRef size, std::optional<ScalarType> dtype_opt, std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt, std::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_cuda(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    fill_empty_deterministic_(result);
  }
  return result;
}

Tensor _efficientzerotensor_cuda(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    auto device_ = device_or_default(device);
    if (!device_.has_index()) {
      device_.set_index(at::cuda::current_device());
    }
    auto allocator = at::native::ZeroTensorAllocator(device_);
    auto dtype_ = dtype_or_default(dtype);
    auto zero_ks = at::DispatchKeySet(c10::DispatchKey::CUDA) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
    auto out = at::detail::empty_generic(size, &allocator, zero_ks, dtype_, std::nullopt);
    return out;
}


Tensor empty_strided_cuda(IntArrayRef size, IntArrayRef stride, std::optional<ScalarType> dtype_opt, std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt) {
  Tensor result = at::detail::empty_strided_cuda(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    fill_empty_deterministic_(result);
  }
  return result;
}

} // namespace at::native
