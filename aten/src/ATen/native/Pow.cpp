#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Pow.h>

#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/result_type.h>
#endif

namespace at::meta {

TORCH_META_FUNC2(pow, Tensor_Tensor) (const Tensor& base, const Tensor& exp) {
  build_borrowing_binary_op(maybe_get_output(), base, exp);
}

TORCH_META_FUNC2(pow, Tensor_Scalar) (const Tensor& base, const Scalar& exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  build_output_borrowing_argument_owning_unary_op(maybe_get_output(), base.to(common_dtype));
}

TORCH_META_FUNC2(pow, Scalar) (const Scalar& base, const Tensor& exp) {
    // This overload doesn't directly use TensorIterator. It attempts to short-circuit,
    // but otherwise redispatches to the Tensor_Tensor overload.
    auto dtype = maybe_get_output().defined() ? maybe_get_output().scalar_type() : at::result_type(base, exp);
    set_output_raw_strided(0, exp.sizes(), {}, exp.options().dtype(dtype), exp.has_names() ? exp.names() : ArrayRef<Dimname>());
}

} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);

TORCH_IMPL_FUNC(pow_Tensor_Tensor_out) (const Tensor& base, const Tensor& exp, const Tensor& out) {
  pow_tensor_tensor_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(pow_Tensor_Scalar_out) (const Tensor& base, const Scalar& exp, const Tensor& out) {
  if (exp.equal(0.0) || exp.equal(false)) {
    out.fill_(1);
  } else if (exp.equal(1.0) || exp.equal(true) ) {
    out.copy_(base);
  } else {
    pow_tensor_scalar_stub(device_type(), *this, exp);
  }
}

TORCH_IMPL_FUNC(pow_Scalar_out) (const Scalar& base, const Tensor& exp, const Tensor& out) {
  if (base.equal(1.0)) {
    out.fill_(1);
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    at::pow_out(const_cast<Tensor&>(out), wrapped_scalar_tensor(base, exp.device()), exp); // redispatch!
  }
}

Tensor& float_power_out(const Tensor& base, const Tensor& exp, Tensor& result) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return result;
}

Tensor& float_power_out(const Tensor& base, const Scalar& exp, Tensor& result) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return result;
}

Tensor& float_power_out(const Scalar& base, const Tensor& exp, Tensor& result) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return result;
}

Tensor float_power(const Tensor& base, const Scalar& exp) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return base;
}

Tensor float_power(const Scalar& base, const Tensor& exp) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return exp;
}

Tensor float_power(const Tensor& base, const Tensor& exp) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return base;
}

Tensor& float_power_(Tensor& base, const Tensor& exp) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return base;
}

Tensor& float_power_(Tensor& base, const Scalar& exp) {
  TORCH_CHECK(false, "float_power is disabled in EasyFHE");
  return base;
}

} // namespace at::native
