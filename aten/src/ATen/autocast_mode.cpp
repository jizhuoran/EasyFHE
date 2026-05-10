#include <ATen/autocast_mode.h>

#include <mutex>
#include <ATen/CachedTensorUtils.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/flat_hash_map.h>

namespace at::autocast {

bool is_autocast_enabled(at::DeviceType device_type) {
  at::DispatchKey dispatch_key = get_autocast_dispatch_key_from_device_type(device_type);
  return !c10::impl::tls_is_dispatch_key_excluded(dispatch_key);
}

void set_autocast_enabled(at::DeviceType device_type, bool enabled) {
  at::DispatchKey dispatch_key = get_autocast_dispatch_key_from_device_type(device_type);
  c10::impl::tls_set_dispatch_key_excluded(dispatch_key, !enabled);
}

namespace {
// Imitate Apex and cache some of the casts to streamline parameter reuse.
// Our heuristic is to cache lower_precision_fp casts of fp32 model weights (see cached_cast below).
//
// After discussion with @ezyang, the cache uses the following structure:
// The key is the fp32 source tensor's TensorImpl*, a proxy for a Tensor uuid that's
// unchanged across shallow copies.
// The value is a tuple with a weakref to the source tensor's TensorImpl as the first
// element and the casted tensor as the second element.
//
// The weakref keeps the source's TensorImpl from being deleted.  We need to because we're
// using the source TensorImpl* as the key.  If it were deleted, another random Tensor could
// be allocated whose TensorImpl* happened to have the same value.  This TensorImpl* would
// then mistakenly hit in cache:  a rare, intermittent, unpredictable bug.
//
// I'm not using the weak_intrusive_ptr as the key because it's more difficult to compare
// directly against incoming TensorImpl*s.
using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, Tensor>;

ska::flat_hash_map<TensorImpl*, val_type>& get_cached_casts() {
  static ska::flat_hash_map<TensorImpl*, val_type> cached_casts;
  return cached_casts;
}
std::mutex cached_casts_mutex;


// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward pass)
// it calls clear_cache() to ensure cached Tensors don't leak outside the autocasting region.
thread_local int nesting = 0;

// The order of this array MUST exactly match the definition order of DeviceType
// in c10/core/DeviceType.h.
static_assert(
    at::COMPILE_TIME_MAX_DEVICE_TYPES == 21,
    "The definition of the default autocast data type per device backend doesn't match with the definition of the device type.");
thread_local std::array<at::ScalarType, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    autocast_dtype = {
        at::kBFloat16, // CPU
        at::kHalf, // CUDA.
        at::ScalarType::Undefined, // Reserved for explicit MKLDNN
        at::ScalarType::Undefined, // OpenGL
        at::ScalarType::Undefined, // OpenCL
        at::ScalarType::Undefined, // IDEEP.
        at::kHalf, // AMD HIP
        at::ScalarType::Undefined, // FPGA
        at::kBFloat16, // ONNX Runtime / Microsoft
        at::kBFloat16, // XLA / TPU
        at::ScalarType::Undefined, // Vulkan
        at::ScalarType::Undefined, // Metal
        at::kHalf, // XPU
        at::kHalf, // MPS
        at::ScalarType::Undefined, // Meta (tensors with no data)
        at::kBFloat16, // HPU / HABANA
        at::ScalarType::Undefined, // SX-Aurora / NEC
        at::ScalarType::Undefined, // Lazy Tensors
        at::kHalf, // Graphcore IPU
        at::kHalf, // Meta training and inference devices
        at::kHalf, // PrivateUse1 device
};

// should we enabled the cache inside autocast.
thread_local bool cache_enabled = true;

} // anonymous namespace

void clear_cache() {
  const std::lock_guard<std::mutex> lock(cached_casts_mutex);
  get_cached_casts().clear();
}

int increment_nesting() {
  return ++nesting;
}

int decrement_nesting() {
  return --nesting;
}

at::ScalarType get_autocast_dtype(at::DeviceType device_type) {
  return autocast_dtype[static_cast<int>(device_type)];
}

void set_autocast_dtype(at::DeviceType device_type, at::ScalarType dtype) {
  autocast_dtype[static_cast<int>(device_type)] = dtype;
}

bool is_autocast_cache_enabled() {
  return cache_enabled;
}

void set_autocast_cache_enabled(bool enabled) {
  cache_enabled = enabled;
}

// Overload to catch Tensor args
// TODO (possible optimization):
// Move cast_cache to an inline function in a header with cached_casts declared as
// extern thread_local in the header.
Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    // Heuristic:  Do what Apex does, and cache lower_precision_fp casts of fp32 model weights (leaves).
    // See cached_casts declaration above for detailed strategy.
    //
    // Fix #158232: Don't cache in inference_mode - those tensors can't be reused
    // for training since enable_grad() cannot override inference_mode.
    bool can_try_cache = (to_type == get_lower_precision_fp_from_device_type(device_type) &&
                         arg.scalar_type() == at::kFloat && arg.requires_grad() &&
                         arg.is_leaf() && !arg.is_view() && cache_enabled &&
                         !at::caching::is_cached_tensor(arg) &&
                         !c10::InferenceMode::is_enabled());

    if (can_try_cache) {
      const std::lock_guard<std::mutex> lock(cached_casts_mutex);
      auto it = get_cached_casts().find(arg.unsafeGetTensorImpl());
      if (it != get_cached_casts().end()) {
        return std::get<1>(it->second);
      } else {
        // Fix #158232: When caching in no_grad() context, we still need grad_fn
        // on the cached tensor so it can be reused in grad-enabled contexts.
        // The AutoGradMode RAII guard temporarily enables grad for .to() only.
        // Note: arg.requires_grad() is guaranteed true here (checked in can_try_cache)
        // and inference_mode is excluded above since enable_grad can't override it.
        c10::AutoGradMode enable_grad(true);
        auto casted_arg = arg.to(to_type);
        get_cached_casts().emplace(arg.unsafeGetTensorImpl(), val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
        return casted_arg;
      }
    } else {
      return arg.to(to_type);
    }
  } else {
    return arg;
  }
}

/*******************************
Banned functions
*******************************/

namespace {

/*****************************************
Explicit registration for out-of-place ops
*****************************************/

TORCH_LIBRARY_IMPL(_, Autocast, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  // lower_precision_fp
#define _KERNEL_CUDA_LOW_PRECISION_FP(...) \
  KERNEL_CUDA(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_CUDA_LOW_PRECISION_FP)

  // fp32
#define _KERNEL_CUDA_FP32(...) KERNEL_CUDA(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_CUDA_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_CUDA_FP32_SET_OPT_DTYPE(...) \
  KERNEL_CUDA(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_CUDA_FP32_SET_OPT_DTYPE)
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL_CUDA(norm, ScalarOpt_dtype, fp32_set_opt_dtype)
  // KERNEL_CUDA(norm, ScalarOpt_dim_dtype, fp32_set_opt_dtype)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA)

  // promote
#define _KERNEL_CUDA_PROMOTE(...) KERNEL_CUDA(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_CUDA_PROMOTE)
}

TORCH_LIBRARY_IMPL(_, AutocastMPS, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastMPS, m) {
  // lower_precision_fp — all deleted (NN/linalg ops)

  // fp32
  KERNEL_MPS(exp, fp32)
  KERNEL_MPS(expm1, fp32)
  KERNEL_MPS(log, fp32)
  KERNEL_MPS(log10, fp32)
  KERNEL_MPS(log2, fp32)
  KERNEL_MPS(log1p, fp32)
  KERNEL_MPS(reciprocal, fp32)
  KERNEL_MPS(rsqrt, fp32)

  // promote
  KERNEL_MPS(addcdiv, promote)
  KERNEL_MPS(addcmul, promote)
  KERNEL_MPS(index_put, promote)
  KERNEL_MPS(scatter_add, promote)
}

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}


TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // lower_precision_fp cast policy — all deleted (NN/linalg ops)

  // fp32 cast policy
  KERNEL_CPU(view_as_complex, fp32)

  // promote
  KERNEL_CPU(stack, promote)
  KERNEL_CPU(cat, promote)
  KERNEL_CPU(index_copy, promote)

}

// MTIA
TORCH_LIBRARY_IMPL(_, AutocastMTIA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastMTIA, m) {
  // lower_precision_fp
#define _KERNEL_MTIA_LOW_PRECISION_FP(...) \
  KERNEL_MTIA(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_MTIA_LOW_PRECISION_FP)

  // fp32
#define _KERNEL_MTIA_FP32(...) KERNEL_MTIA(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_MTIA_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_MTIA_FP32_SET_OPT_DTYPE(...) \
  KERNEL_MTIA(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_MTIA_FP32_SET_OPT_DTYPE)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_MTIA)

  // promote
#define _KERNEL_MTIA_PROMOTE(...) KERNEL_MTIA(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_MTIA_PROMOTE)
}

// MAIA
TORCH_LIBRARY_IMPL(_, AutocastMAIA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastMAIA, m) {
  // lower_precision_fp
#define _KERNEL_MAIA_LOW_PRECISION_FP(...) \
  KERNEL_MAIA(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_MAIA_LOW_PRECISION_FP)

  // fp32
#define _KERNEL_MAIA_FP32(...) KERNEL_MAIA(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_MAIA_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_MAIA_FP32_SET_OPT_DTYPE(...) \
  KERNEL_MAIA(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_MAIA_FP32_SET_OPT_DTYPE)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_MAIA)

  // promote
#define _KERNEL_MAIA_PROMOTE(...) KERNEL_MAIA(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_MAIA_PROMOTE)
}

// XPU
TORCH_LIBRARY_IMPL(_, AutocastXPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastXPU, m) {
  // lower_precision_fp
#define _KERNEL_XPU_LOW_PRECISION_FP(...) \
  KERNEL_XPU(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_XPU_LOW_PRECISION_FP)

  // fp32
#define _KERNEL_XPU_FP32(...) KERNEL_XPU(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_XPU_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_XPU_FP32_SET_OPT_DTYPE(...) \
  KERNEL_XPU(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_XPU_FP32_SET_OPT_DTYPE)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XPU)

  // promote
#define _KERNEL_XPU_PROMOTE(...) KERNEL_XPU(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_XPU_PROMOTE)
}

} // namespace
} // namespace at::autocast
