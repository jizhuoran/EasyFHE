#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <type_traits>
#include <utility>

#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorMeta.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/copysign.h>
#include <ATen/ops/copysign_native.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/fmod.h>
#include <ATen/ops/fmod_native.h>
#include <ATen/ops/full.h>
#include <ATen/ops/heaviside_native.h>
#include <ATen/ops/hypot_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/multiply_native.h>
#include <ATen/ops/nextafter_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/remainder_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/subtract_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

TORCH_META_FUNC2(sub, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  native::sub_check(self, other);
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

TORCH_META_FUNC2(mul, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor_mode) (const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    build_borrowing_binary_float_op(maybe_get_output(), self, other);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (*rounding_mode == "trunc") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else if (*rounding_mode == "floor") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else {
    TORCH_CHECK(false,
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        "but found '", *rounding_mode, "'");
  }
}

TORCH_META_FUNC2(copysign, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(heaviside) (
  const Tensor& self, const Tensor& other
) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex() &&
              (maybe_get_output().defined() ? !maybe_get_output().is_complex() : true),
              "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(self.dtype() == other.dtype() &&
              (maybe_get_output().defined() ? maybe_get_output().dtype() == self.dtype() : true),
              "heaviside is not yet implemented for tensors with different dtypes.");

  build_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(remainder, Tensor)(const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(fmod, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

// These are normal binary ops that preserve dtype
#define CREATE_BINARY_META_FUNC(func)                                 \
  TORCH_META_FUNC(func) (const Tensor& self, const Tensor& other) {   \
    build_borrowing_binary_op(maybe_get_output(), self, other);                 \
  }

CREATE_BINARY_META_FUNC(hypot)
CREATE_BINARY_META_FUNC(nextafter)

#define CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(func)                     \
  TORCH_META_FUNC2(func, Tensor)(const Tensor& self, const Tensor& other) { \
    const Tensor& result = maybe_get_output();                              \
    build_borrowing_comparison_op(result, self, other);                     \
  }                                                                         \
                                                                            \
  TORCH_META_FUNC2(func, Scalar)(const Tensor& self, const Scalar& other) { \
    auto other_tensor =                                                     \
        native::wrapped_scalar_tensor(other);                               \
    build_borrowing_except_last_argument_comparison_op(maybe_get_output(), self, other_tensor);  \
  }


} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(add_clamp_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(div_true_stub);
DEFINE_DISPATCH(div_floor_stub);
DEFINE_DISPATCH(div_trunc_stub);
DEFINE_DISPATCH(remainder_stub);
DEFINE_DISPATCH(atan2_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);
DEFINE_DISPATCH(lshift_stub);
DEFINE_DISPATCH(rshift_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);
DEFINE_DISPATCH(logical_xor_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(maximum_stub);
DEFINE_DISPATCH(minimum_stub);
DEFINE_DISPATCH(fmax_stub);
DEFINE_DISPATCH(fmin_stub);
DEFINE_DISPATCH(fmod_stub);
DEFINE_DISPATCH(logaddexp_stub);
DEFINE_DISPATCH(logaddexp2_stub);
DEFINE_DISPATCH(gcd_stub);
DEFINE_DISPATCH(lcm_stub);
DEFINE_DISPATCH(hypot_stub);
DEFINE_DISPATCH(nextafter_stub);
DEFINE_DISPATCH(heaviside_stub);
DEFINE_DISPATCH(copysign_stub);
DEFINE_DISPATCH(zeta_stub);
DEFINE_DISPATCH(chebyshev_polynomial_t_stub);
DEFINE_DISPATCH(chebyshev_polynomial_u_stub);
DEFINE_DISPATCH(chebyshev_polynomial_v_stub);
DEFINE_DISPATCH(chebyshev_polynomial_w_stub);
DEFINE_DISPATCH(hermite_polynomial_h_stub);
DEFINE_DISPATCH(hermite_polynomial_he_stub);
DEFINE_DISPATCH(laguerre_polynomial_l_stub);
DEFINE_DISPATCH(legendre_polynomial_p_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_t_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_u_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_v_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_w_stub);
DEFINE_DISPATCH(ldexp_stub);
DEFINE_DISPATCH(huber_stub);
DEFINE_DISPATCH(igamma_stub);
DEFINE_DISPATCH(igammac_stub);
DEFINE_DISPATCH(logit_backward_stub);
DEFINE_DISPATCH(max_elementwise_stub);
DEFINE_DISPATCH(min_elementwise_stub);
DEFINE_DISPATCH(mse_stub);
DEFINE_DISPATCH(smooth_l1_stub);
DEFINE_DISPATCH(xlog1py_stub);
DEFINE_DISPATCH(xlogy_stub);

TORCH_IMPL_FUNC(sub_out) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  add_stub(device_type(), *this, -alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}

TORCH_IMPL_FUNC(mul_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  mul_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  div_true_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out_mode) (
  const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode, const Tensor& result
) {
  if (!rounding_mode.has_value()) {
    div_true_stub(device_type(), *this);
  } else if (*rounding_mode == "trunc") {
    div_trunc_stub(device_type(), *this);
  } else if (*rounding_mode == "floor") {
    div_floor_stub(device_type(), *this);
  }
}

#define CREATE_BINARY_TORCH_IMPL_FUNC(func_out, func_stub)                                                    \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& other, const Tensor& result) {  \
  func_stub(device_type(), *this);                                                           \
}

CREATE_BINARY_TORCH_IMPL_FUNC(fmod_out, fmod_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(remainder_out, remainder_stub)

Tensor arctan2(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "atan2/arctan2 is disabled in EasyFHE");
  return self;
}

Tensor& arctan2_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "atan2/arctan2 is disabled in EasyFHE");
  return self;
}

Tensor& arctan2_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "atan2/arctan2 is disabled in EasyFHE");
  return result;
}

TORCH_IMPL_FUNC(copysign_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  copysign_stub(device_type(), *this);
}

Tensor copysign(const Tensor& self, const Scalar& other) {
  // redispatch!
  return at::copysign(self, wrapped_scalar_tensor(other));
}

Tensor& copysign_(Tensor& self, const Scalar& other) {
  // redispatch!
  return self.copysign_(wrapped_scalar_tensor(other));
}

Tensor& copysign_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch!
  return at::copysign_out(result, self, wrapped_scalar_tensor(other));
}

// WARNING: There doesn't appear to be any testing for this function
// with sparse self input.
Tensor div(const Tensor& self, const Scalar& other) {
  return self.div(wrapped_scalar_tensor(other)); // redispatch!
}

// WARNING: This function, with a sparse self, is currently only
// exercised by DistributedDataParallelTest.test_sparse_gradients
// (you need to exercise it from C++, because this overload is never
// used for Python)
Tensor& div_(Tensor& self, const Scalar& other) {
  return self.div_(wrapped_scalar_tensor(other)); // redispatch!
}

Tensor div(const Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
  return self.div(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

Tensor& div_(Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
  return self.div_(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

// divide, alias for div
Tensor& divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::div_out(result, self, other);
}

Tensor divide(const Tensor& self, const Tensor& other) {
  return self.div(other);
}

Tensor& divide_(Tensor& self, const Tensor& other) {
  return self.div_(other);
}

Tensor divide(const Tensor& self, const Scalar& other) {
  return self.div(other);
}

Tensor& divide_(Tensor& self, const Scalar& other) {
  return self.div_(other);
}

Tensor& divide_out(const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode, Tensor& result) {
  return at::div_out(result, self, other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

// true_divide, an alias for div
Tensor& true_divide_out(const Tensor& self, const Tensor& divisor, Tensor& result) {
  return at::div_out(result, self, divisor);
}

Tensor true_divide(const Tensor& self, const Tensor& divisor) {
  return self.div(divisor);
}

Tensor& true_divide_(Tensor& self, const Tensor& divisor) {
  return self.div_(divisor);
}

Tensor true_divide(const Tensor& self, const Scalar& divisor) {
  return self.div(divisor);
}

Tensor& true_divide_(Tensor& self, const Scalar& divisor) {
  return self.div_(divisor);
}

Tensor& floor_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  div_floor_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor floor_divide(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  div_floor_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return native::floor_divide_out(self, other, self);
}

// TODO: Make this structured to undo the perf regression from native:: removal
// in call here
Tensor mul(const Tensor& self, const Scalar& other) {
  return at::mul(self, wrapped_scalar_tensor(other)); // redispatch!
}

Tensor& mul_(Tensor& self, const Scalar& other) {
  return at::mul_out(self, wrapped_scalar_tensor(other), self); // redispatch!
}

Tensor& mul__scalar_sparse_csr(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "Sparse tensors not supported in EasyFHE");
}

static Device correct_out_device(const Tensor& self, const Tensor& other) {
  if (self.device() == at::kCPU){
      return other.device();
  } else {
    return self.device();
  }
}

static Tensor send_to_meta(const Tensor& self, const Device& device) {
  Tensor out_meta;
  if (self._is_zerotensor() && self.unsafeGetTensorImpl()->is_wrapped_number()) {
    out_meta = at::_efficientzerotensor(self.sizes(), self.options().device(device));
    out_meta.unsafeGetTensorImpl()->set_wrapped_number(true);
  } else {
    out_meta = self.to(device);
  }
  return out_meta;
}

Tensor mul_zerotensor(const Tensor& self, const Tensor& other) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto self_meta = send_to_meta(self, device_);
  auto other_meta = send_to_meta(other, device_);
  auto meta_out = at::_ops::mul_Tensor::redispatch(meta_dks, self_meta, other_meta);
  return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
}

Tensor div_zerotensor(const Tensor& self, const Tensor& other) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto self_meta = send_to_meta(self, device_);
  auto other_meta = send_to_meta(other, device_);
  auto meta_out = at::_ops::div_Tensor::redispatch(meta_dks, self_meta, other_meta);

  if (self._is_zerotensor()) {
    if (other._is_zerotensor()) {
      // 0/0, return full NAN
      return at::full(meta_out.sizes(), std::numeric_limits<float>::quiet_NaN(), meta_out.options().device(out_device));
    }
    else {
      // 0/x, return zero tensor
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
  }
  else {
    if (other._is_zerotensor()) {
      // x/0, return full INF
      return at::full(meta_out.sizes(), std::numeric_limits<float>::infinity(), meta_out.options().device(out_device));
    }
    else {
      // x/y -- unreachable, see TORCH_INTERNAL_ASSERT above
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
  }
}

static Tensor maybe_add_maybe_sub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto self_meta = send_to_meta(self, device_);
  auto other_meta = send_to_meta(other, device_);
  auto meta_out = at::_ops::add_Tensor::redispatch(meta_dks, self_meta, other_meta, alpha);

  auto get_out_like = [&] (const Tensor& tensor)
  {
      auto sizes = meta_out.sizes();
      return at::_to_copy(tensor.expand(sizes), meta_out.options().device(out_device));
  };

  if (self._is_zerotensor()) {
    if (other._is_zerotensor()) {
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
    auto res = get_out_like(other);
    return alpha.equal(1) ? std::move(res) : res.mul(alpha);
  } else {
    return get_out_like(self);
  }
}
Tensor add_zerotensor(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return maybe_add_maybe_sub(self, other, alpha);
}

Tensor sub_zerotensor(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return maybe_add_maybe_sub(self, other, -alpha);
}

// multiply, alias for mul
Tensor& multiply_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::mul_out(result, self, other);
}

Tensor multiply(const Tensor& self, const Tensor& other) {
  return self.mul(other);
}

Tensor& multiply_(Tensor& self, const Tensor& other) {
  return self.mul_(other);
}

Tensor multiply(const Tensor& self, const Scalar& other) {
  return self.mul(other);
}

Tensor& multiply_(Tensor& self, const Scalar& other) {
  return self.mul_(other);
}

Tensor sub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::sub(self, wrapped_scalar_tensor(other), alpha); // redispatch!
}

Tensor& sub_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(wrapped_scalar_tensor(other), alpha); // redispatch!
}

// subtract, alias for sub
Tensor& subtract_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
  return at::sub_out(result, self, other, alpha);
}

Tensor subtract(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

Tensor& subtract_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

Tensor subtract(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

Tensor& subtract_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::sub(other, self, alpha); // redispatch!
}

// TODO: Make this structured to undo the perf regression from native:: removal
// in call here

Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.add_(wrapped_scalar_tensor(other), alpha);
}

Tensor remainder(const Tensor& self, const Scalar& other) {
  // redispatch
  return at::remainder(self, wrapped_scalar_tensor(other));
}

Tensor& remainder_(Tensor& self, const Scalar& other) {
  // redispatch
  return self.remainder_(wrapped_scalar_tensor(other));
}

Tensor& remainder_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch
  return at::remainder_out(result, self, wrapped_scalar_tensor(other));
}

Tensor remainder(const Scalar& self, const Tensor& other) {
  return at::remainder(wrapped_scalar_tensor(self), other);
}

Tensor rsub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& bitwise_and_out(const Tensor& self, const Scalar& other, Tensor& result) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return result;
}

Tensor bitwise_and(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return self;
}

Tensor bitwise_and(const Scalar& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return other;
}

Tensor& bitwise_and_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return self;
}

// Legacy and interfaces. They are aliased to bitwise_and* functions
Tensor __and__(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return self;
}

Tensor __and__(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return self;
}

Tensor& __iand__(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return self;
}

Tensor& __iand__(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_and is disabled in EasyFHE");
  return self;
}

Tensor& bitwise_or_out(const Tensor& self, const Scalar& other, Tensor& result) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return result;
}

Tensor bitwise_or(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return self;
}

Tensor bitwise_or(const Scalar& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return other;
}

Tensor& bitwise_or_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return self;
}

// Legacy or interfaces. They are aliased to bitwise_or* functions
Tensor __or__(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return self;
}

Tensor __or__(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return self;
}

Tensor& __ior__(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return self;
}

Tensor& __ior__(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_or is disabled in EasyFHE");
  return self;
}

Tensor& bitwise_xor_out(const Tensor& self, const Scalar& other, Tensor& result) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return result;
}

Tensor bitwise_xor(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return self;
}

Tensor bitwise_xor(const Scalar& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return other;
}

Tensor& bitwise_xor_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return self;
}

// Legacy xor interfaces. They are aliased to bitwise_xor* functions
Tensor __xor__(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return self;
}

Tensor __xor__(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return self;
}

Tensor& __ixor__(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return self;
}

Tensor& __ixor__(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_xor is disabled in EasyFHE");
  return self;
}

Tensor __lshift__(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_left_shift is disabled in EasyFHE");
  return self;
}

Tensor __lshift__(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_left_shift is disabled in EasyFHE");
  return self;
}

Tensor& __ilshift__(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_left_shift is disabled in EasyFHE");
  return self;
}

Tensor& __ilshift__(Tensor& self, const Scalar& other) {
  return self;
}

Tensor& bitwise_left_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  TORCH_CHECK(false, "bitwise_left_shift is disabled in EasyFHE");
  return result;
}

Tensor bitwise_left_shift(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_left_shift is disabled in EasyFHE");
  return self;
}

Tensor& bitwise_left_shift_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_left_shift is disabled in EasyFHE");
  return self;
}

Tensor bitwise_left_shift(const Scalar& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_left_shift is disabled in EasyFHE");
  return other;
}

Tensor __rshift__(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return self;
}

Tensor __rshift__(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return self;
}

Tensor& __irshift__(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return self;
}

Tensor& __irshift__(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return self;
}

Tensor& bitwise_right_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return result;
}

Tensor bitwise_right_shift(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return self;
}

Tensor& bitwise_right_shift_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return self;
}

Tensor bitwise_right_shift(const Scalar& self, const Tensor& other) {
  TORCH_CHECK(false, "bitwise_right_shift is disabled in EasyFHE");
  return other;
}

template <typename Stub>
static Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Tensor& other, Stub& stub) {
  auto iter = TensorIterator::comparison_op(result, self, other);
  stub(iter.device_type(), iter);
  return result;
}

template <typename OutImpl>
static Tensor comparison_op(const Tensor& self, const Tensor& other, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return out_impl(result, self, other);
}

template <typename OutImpl>
static Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  return out_impl(self, self, other);
}

template <typename OutImpl>
static Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(result, self, wrapped_scalar_tensor(other));
}

template <typename OutImpl>
static Tensor comparison_op(const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return comparison_op(self, wrapped_scalar_tensor(other), out_impl);
}

template <typename OutImpl>
static Tensor& comparison_op_(Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(self, self, wrapped_scalar_tensor(other));
}

// We need explicit cast to OutFunc because each *_out func is overloaded twice. Without An explicit cast, merely
// referring to *_out function is ambiguous.
using OutFunc = std::add_const_t<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>;

// less, alias for torch.lt
Tensor& less_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "less is disabled in EasyFHE"); return result; }
Tensor less(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "less is disabled in EasyFHE"); return self; }
Tensor& less_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "less is disabled in EasyFHE"); return self; }
Tensor& less_out(const Tensor& self, const Scalar& other, Tensor& result) { TORCH_CHECK(false, "less is disabled in EasyFHE"); return result; }
Tensor less(const Tensor& self, const Scalar& other) { TORCH_CHECK(false, "less is disabled in EasyFHE"); return self; }
Tensor& less_(Tensor& self, const Scalar& other) { TORCH_CHECK(false, "less is disabled in EasyFHE"); return self; }

// less_equal, alias for torch.le
Tensor& less_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "less_equal is disabled in EasyFHE"); return result; }
Tensor less_equal(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "less_equal is disabled in EasyFHE"); return self; }
Tensor& less_equal_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "less_equal is disabled in EasyFHE"); return self; }
Tensor& less_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { TORCH_CHECK(false, "less_equal is disabled in EasyFHE"); return result; }
Tensor less_equal(const Tensor& self, const Scalar& other) { TORCH_CHECK(false, "less_equal is disabled in EasyFHE"); return self; }
Tensor& less_equal_(Tensor& self, const Scalar& other) { TORCH_CHECK(false, "less_equal is disabled in EasyFHE"); return self; }

// greater, alias for torch.gt
Tensor& greater_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "greater is disabled in EasyFHE"); return result; }
Tensor greater(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "greater is disabled in EasyFHE"); return self; }
Tensor& greater_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "greater is disabled in EasyFHE"); return self; }
Tensor& greater_out(const Tensor& self, const Scalar& other, Tensor& result) { TORCH_CHECK(false, "greater is disabled in EasyFHE"); return result; }
Tensor greater(const Tensor& self, const Scalar& other) { TORCH_CHECK(false, "greater is disabled in EasyFHE"); return self; }
Tensor& greater_(Tensor& self, const Scalar& other) { TORCH_CHECK(false, "greater is disabled in EasyFHE"); return self; }

// greater_equal, alias for torch.ge
Tensor& greater_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "greater_equal is disabled in EasyFHE"); return result; }
Tensor greater_equal(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "greater_equal is disabled in EasyFHE"); return self; }
Tensor& greater_equal_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "greater_equal is disabled in EasyFHE"); return self; }
Tensor& greater_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { TORCH_CHECK(false, "greater_equal is disabled in EasyFHE"); return result; }
Tensor greater_equal(const Tensor& self, const Scalar& other) { TORCH_CHECK(false, "greater_equal is disabled in EasyFHE"); return self; }
Tensor& greater_equal_(Tensor& self, const Scalar& other) { TORCH_CHECK(false, "greater_equal is disabled in EasyFHE"); return self; }

// not_equal, alias for torch.ne
Tensor& not_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "not_equal is disabled in EasyFHE"); return result; }
Tensor not_equal(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "not_equal is disabled in EasyFHE"); return self; }
Tensor& not_equal_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "not_equal is disabled in EasyFHE"); return self; }
Tensor& not_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { TORCH_CHECK(false, "not_equal is disabled in EasyFHE"); return result; }
Tensor not_equal(const Tensor& self, const Scalar& other) { TORCH_CHECK(false, "not_equal is disabled in EasyFHE"); return self; }
Tensor& not_equal_(Tensor& self, const Scalar& other) { TORCH_CHECK(false, "not_equal is disabled in EasyFHE"); return self; }

Tensor& logical_and_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "logical_and is disabled in EasyFHE"); return result; }
Tensor logical_and(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "logical_and is disabled in EasyFHE"); return self; }
Tensor& logical_and_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "logical_and is disabled in EasyFHE"); return self; }

Tensor& logical_or_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "logical_or is disabled in EasyFHE"); return result; }
Tensor logical_or(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "logical_or is disabled in EasyFHE"); return self; }
Tensor& logical_or_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "logical_or is disabled in EasyFHE"); return self; }

Tensor& logical_xor_out(const Tensor& self, const Tensor& other, Tensor& result) { TORCH_CHECK(false, "logical_xor is disabled in EasyFHE"); return result; }
Tensor logical_xor(const Tensor& self, const Tensor& other) { TORCH_CHECK(false, "logical_xor is disabled in EasyFHE"); return self; }
Tensor& logical_xor_(Tensor& self, const Tensor& other) { TORCH_CHECK(false, "logical_xor is disabled in EasyFHE"); return self; }

// binary max, alias for maximum
Tensor& max_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "maximum/max Tensor overload is disabled in EasyFHE");
  return result;
}

Tensor max(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "maximum/max Tensor overload is disabled in EasyFHE");
  return self;
}

// binary min, alias for minimum
Tensor& min_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "minimum/min Tensor overload is disabled in EasyFHE");
  return result;
}

Tensor min(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "minimum/min Tensor overload is disabled in EasyFHE");
  return self;
}

Tensor floor_divide(const Tensor& self, const Scalar& other) {
  return at::floor_divide(self, wrapped_scalar_tensor(other));
}

Tensor& floor_divide_(Tensor& self, const Scalar& other) {
  return at::floor_divide_out(self, self, wrapped_scalar_tensor(other));
}

Tensor& fmod_out(const Tensor& self, const Scalar& other, Tensor & result) {
  // redispatch
  return at::fmod_out(result, self, wrapped_scalar_tensor(other));
}

Tensor fmod(const Tensor& self, const Scalar& other) {
  // redispatch
  return at::fmod(self, wrapped_scalar_tensor(other));
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  // redispatch
  return self.fmod_(wrapped_scalar_tensor(other));
}

// Note: this function is only for testing.
// It is undocumented and should not be used outside of tests.

TORCH_IMPL_FUNC(heaviside_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  heaviside_stub(device_type(), *this);
}

static inline Tensor _pow2(const Tensor& self, const Tensor& other) {
  const auto self_dtype = self.scalar_type();
  // All integral types are promoted to float32
  if (isIntegralType(self_dtype, true) || self_dtype == kFloat) {
      return at::pow(2.0, other);
  }
  // For double and reduced floating types do regular type promotion
  return at::full({}, 2.0, self.options()).pow(other);
}

// This function is used to dispatch to kernels that use std::ldexp on CPU and the global namespaces ::ldexp on CUDA
// Both of these require floating types for 'self' and integer types for 'other'.
static inline Tensor& _ldexp_int_exponent(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(result)
    .add_input(self)
    .add_input(other)
    .build();

  ldexp_stub(iter.device_type(), iter);
  return result;
}

Tensor& ldexp_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "ldexp is disabled in EasyFHE");
  return result;
}

Tensor ldexp(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "ldexp is disabled in EasyFHE");
  return self;
}

Tensor& ldexp_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "ldexp is disabled in EasyFHE");
  return self;
}

} // namespace at::native
