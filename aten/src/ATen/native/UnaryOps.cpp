#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>

#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/ComplexHelper.h>

#include <c10/util/MathConstants.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conj_native.h>
#include <ATen/ops/_conj_physical.h>
#include <ATen/ops/_conj_physical_native.h>
#include <ATen/ops/_neg_view_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/angle_native.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/can_cast.h>
#include <ATen/ops/conj_native.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exp_native.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/imag_native.h>
#include <ATen/ops/log10_native.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/log2_native.h>
#include <ATen/ops/log_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/nan_to_num_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/negative_native.h>
#include <ATen/ops/positive_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/real_native.h>
#include <ATen/ops/reciprocal_native.h>
#include <ATen/ops/resolve_conj_native.h>
#include <ATen/ops/resolve_neg_native.h>
#include <ATen/ops/rsqrt_native.h>
#include <ATen/ops/select.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/square_native.h>
#include <ATen/ops/view_as_real.h>
#endif

#include <cmath>

namespace at::meta {

// Unary float operations always produce floating point
// outputs for floating point and integral types
// For complex inputs, the output type should be the same as input type.
#define CREATE_UNARY_FLOAT_META_FUNC(func)                  \
  TORCH_META_FUNC(func) (const Tensor& self) {        \
    build_borrowing_unary_float_op(maybe_get_output(), self);   \
  }

CREATE_UNARY_FLOAT_META_FUNC(exp)
CREATE_UNARY_FLOAT_META_FUNC(expm1)
CREATE_UNARY_FLOAT_META_FUNC(log)
CREATE_UNARY_FLOAT_META_FUNC(log10)
CREATE_UNARY_FLOAT_META_FUNC(log1p)
CREATE_UNARY_FLOAT_META_FUNC(log2)
CREATE_UNARY_FLOAT_META_FUNC(reciprocal)
CREATE_UNARY_FLOAT_META_FUNC(rsqrt)
CREATE_UNARY_FLOAT_META_FUNC(sqrt)


// These are normal unary ops that preserve dtype
#define CREATE_UNARY_META_FUNC(func)                  \
  TORCH_META_FUNC(func) (const Tensor& self) {        \
    build_borrowing_unary_op(maybe_get_output(), self);   \
  }
CREATE_UNARY_META_FUNC(sgn)

TORCH_META_FUNC(neg)(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  build_borrowing_unary_op(maybe_get_output(), self);
}

} // namespace at::meta

namespace at::native {
// NOTE: These are helper functions that reduce redundant code in implementing the most typical kind of unary operators.
// YOU ARE NOT OBLIGED TO USE THESE HELPERS---if you're writing something more specialized, please don't try to make
// them work for your case, but just write something new instead. Here we use helper functions instead of a flat fat
// macro that implements everything, because the former allows some simple preprocessing that are unique to some
// operators (more is foreseeable) and is more flexible and elegant than the latter.
#define CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  func_stub(device_type(), *this);                                      \
}

// This macro is as optional as the one above. torch.(ceil|floor|round|trunc) are no-ops for integers
// See gh-70918
#define CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {                                      \
    result.copy_(self);                                                 \
  } else {                                                              \
    func_stub(device_type(), *this);                                    \
  }                                                                     \
}
CREATE_UNARY_TORCH_IMPL_FUNC(exp_out, exp_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(expm1_out, expm1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log10_out, log10_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log1p_out, log1p_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log2_out, log2_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)

template <typename Stub>
static inline Tensor& unary_op_impl_out(Tensor& result, const Tensor& self, Stub& stub) {
  auto iter = TensorIterator::unary_op(result, self);
  stub(iter.device_type(), iter);
  return result;
}

template <typename Stub, typename ...Args>
static inline Tensor& unary_op_impl_float_out(Tensor& result, const Tensor& self, Stub& stub, Args... args) {
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter.device_type(), iter, args...);
  return result;
}

template <typename Stub, typename ...Args>
static inline Tensor unary_op_impl_float(const Tensor& self, Stub& stub, Args... args) {
  Tensor result;
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter.device_type(), iter, args...);
  return iter.output();
}

// An alternate version of unary_op_impl_out that follows the same pattern
// for non-complex inputs, but returns a floating point tensor
// for complex inputs by default.
// Note: This is done by running the operation as usual and then copying the
// operation's result to the expected result type.
template <typename Stub>
static inline Tensor& unary_op_impl_with_complex_to_float_out(Tensor& result, const Tensor& self, Stub& stub, bool promotes_integer_to_float) {
    if (self.is_complex() && !result.is_complex()) {
      // Checks if the corresponding float type can be cast to the desired dtype
      const auto float_type = c10::toRealValueType(self.scalar_type());
      TORCH_CHECK(canCast(float_type, result.scalar_type()),
            "result type ", float_type, " can't be cast to the desired output type ",
            result.scalar_type());

      // Runs the function complex->complex, as TensorIterator expects
      Tensor complex_result = at::empty({0}, self.options());
      auto iter = TensorIterator::unary_op(complex_result, self);
      stub(iter.device_type(), iter);

      // Copies the complex result to the actual result and returns it
      at::native::resize_output(result, complex_result.sizes());
      result.copy_(at::real(complex_result));
      return result;
    }

    if (promotes_integer_to_float) {
      return unary_op_impl_float_out(result, self, stub);
    }

    return unary_op_impl_out(result, self, stub);
}

// out_impl passed into unary_op_impl and unary_op_impl_  must go through at:: device dispatch
// otherwise it won't dispatch to out-of-source devices like XLA.
// For example it must be at::bitwise_not_out instead of bitwise_not_out(which is at::native!).
template <typename OutImpl>
static inline Tensor unary_op_impl(const Tensor& self, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

// An alternate version of unary_op_impl that follows the same pattern
// for non-complex inputs, but returns a floating point tensor
// for complex inputs by default.
template <typename OutImpl>
static inline Tensor unary_op_impl_with_complex_to_float(const Tensor& self, OutImpl& out_impl) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty_like(self, self.options().dtype(float_type));
    return out_impl(result, self);
  }

  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  return out_impl(self, self);
}

// arccos, alias for acos
Tensor& arccos_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "arccos is disabled in EasyFHE");
  return result;
}
Tensor arccos(const Tensor& self) {
  TORCH_CHECK(false, "arccos is disabled in EasyFHE");
  return self;
}
Tensor& arccos_(Tensor& self) {
  TORCH_CHECK(false, "arccos is disabled in EasyFHE");
  return self;
}

Tensor& rad2deg_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "rad2deg is disabled in EasyFHE");
  return result;
}
Tensor rad2deg(const Tensor& self) {
  TORCH_CHECK(false, "rad2deg is disabled in EasyFHE");
  return self;
}
Tensor& rad2deg_(Tensor& self) {
  TORCH_CHECK(false, "rad2deg is disabled in EasyFHE");
  return self;
}

Tensor& deg2rad_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "deg2rad is disabled in EasyFHE");
  return result;
}
Tensor deg2rad(const Tensor& self) {
  TORCH_CHECK(false, "deg2rad is disabled in EasyFHE");
  return self;
}
Tensor& deg2rad_(Tensor& self) {
  TORCH_CHECK(false, "deg2rad is disabled in EasyFHE");
  return self;
}

// arcsin, alias of asin
Tensor& arcsin_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "arcsin is disabled in EasyFHE");
  return result;
}
Tensor arcsin(const Tensor& self) {
  TORCH_CHECK(false, "arcsin is disabled in EasyFHE");
  return self;
}
Tensor& arcsin_(Tensor& self) {
  TORCH_CHECK(false, "arcsin is disabled in EasyFHE");
  return self;
}

// arctan, alias of atan
Tensor& arctan_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "arctan is disabled in EasyFHE");
  return result;
}
Tensor arctan(const Tensor& self) {
  TORCH_CHECK(false, "arctan is disabled in EasyFHE");
  return self;
}
Tensor& arctan_(Tensor& self) {
  TORCH_CHECK(false, "arctan is disabled in EasyFHE");
  return self;
}

// Note [Complex abs and angle]
// Complex inputs to abs and angle return float results by default.
// abs and angle, in both NumPy and C++, returns a float result when given a
// complex input. This makes sense mathematically since the absolute value
// and angle of a complex number has no imaginary part.
Tensor& abs_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_with_complex_to_float_out(result, self, abs_stub, /*promotes_integer_to_float=*/false);
}
Tensor abs(const Tensor& self) {
  return unary_op_impl_with_complex_to_float(self, at::abs_out);
}
Tensor& abs_(Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "In-place abs is not supported for complex tensors.");
  return unary_op_impl_(self, at::abs_out);
}

// Absolute, alias for abs
Tensor& absolute_out(const Tensor& self, Tensor& result) {
  return at::abs_out(result, self);
}
Tensor absolute(const Tensor& self) {
  return self.abs();
}
Tensor& absolute_(Tensor& self) {
  return self.abs_();
}

Tensor& angle_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_with_complex_to_float_out(result, self, angle_stub, /*promotes_integer_to_float=*/true);
}
Tensor angle(const Tensor& self) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty({0}, self.options().dtype(float_type));
    return at::angle_out(result, self);
  }

  return unary_op_impl_float(self, angle_stub);
}

Tensor real(const Tensor& self) {
  if (self.is_complex()) {
    Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = at::view_as_real(self._conj());
    } else {
      real_tensor = at::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 0);
  } else {
    return self;
  }
}

Tensor _neg_view(const Tensor& self) {
  Tensor self_ = self.alias();
  self_._set_neg(!self.is_neg());
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor imag(const Tensor& self) {
  if (self.is_complex()) {
    Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = at::view_as_real(self._conj());
      // preemptively set the negative flag for the final imag tensor
      real_tensor = real_tensor._neg_view();
    } else {
      real_tensor = at::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 1);
  } else {
    TORCH_CHECK(false, "imag is not implemented for tensors with non-complex dtypes.");
  }
}

Tensor& conj_physical_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_out(result, self, conj_physical_stub);
}

Tensor _conj_physical(const Tensor& self) {
  if (self.is_conj()) {
    return self.conj().clone();
  }
  auto result = at::empty_like(self);
  return at::conj_physical_out(result, self);
}

Tensor conj_physical(const Tensor& self) {
  if (!self.is_complex()) return self;
  return at::_conj_physical(self);
}

Tensor& conj_physical_(Tensor& self) {
  if (!self.is_complex()) return self;
  return unary_op_impl_out(self, self, conj_physical_stub);
}

// No op if the neg bit is not set
// else returns a new negated tensor with neg bit set to 0
Tensor resolve_neg(const Tensor& self) {
  if (!self.is_neg()) { return self; }
  // negation is materialized in `copy_()` that clone ultimately calls into
  return self.clone();
}

// No op if the conj bit is not set
// else returns a new negated tensor with neg bit set to 0
Tensor resolve_conj(const Tensor& self) {
  if (!self.is_conj()) { return self; }
  // conjugation is materialized in `copy_()` that clone ultimately calls into
  return self.clone();
}

Tensor _conj(const Tensor& self) {
  Tensor self_ = self.alias();
  self_._set_conj(!self.is_conj());
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor conj(const Tensor& self) {
  // This might look like an infinite recursion but it's not.
  // This actually calls into `conj()` defined in the Tensor class.
  return self.conj();
}


// FIXME: remove const_cast once unary_op_impl_out is updated
TORCH_IMPL_FUNC(sgn_out) (const Tensor& self, const Tensor& result) {
  if (self.is_complex()) {
    sgn_stub(device_type(), *this);
  } else {
    sign_stub(device_type(), *this);
  }
}

// arccosh, alias for acosh
Tensor& arccosh_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "arccosh is disabled in EasyFHE");
  return result;
}
Tensor arccosh(const Tensor& self) {
  TORCH_CHECK(false, "arccosh is disabled in EasyFHE");
  return self;
}
Tensor& arccosh_(Tensor& self) {
  TORCH_CHECK(false, "arccosh is disabled in EasyFHE");
  return self;
}

// arcsinh, alias for asinh
Tensor& arcsinh_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "arcsinh is disabled in EasyFHE");
  return result;
}
Tensor arcsinh(const Tensor& self) {
  TORCH_CHECK(false, "arcsinh is disabled in EasyFHE");
  return self;
}
Tensor& arcsinh_(Tensor& self) {
  TORCH_CHECK(false, "arcsinh is disabled in EasyFHE");
  return self;
}

// arctanh, alias for atanh
Tensor& arctanh_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "arctanh is disabled in EasyFHE");
  return result;
}
Tensor arctanh(const Tensor& self) {
  TORCH_CHECK(false, "arctanh is disabled in EasyFHE");
  return self;
}
Tensor& arctanh_(Tensor& self) {
  TORCH_CHECK(false, "arctanh is disabled in EasyFHE");
  return self;
}

Tensor& square_out(const Tensor& self, Tensor& result) { return at::mul_out(result, self, self); }
Tensor square(const Tensor& self) { return self.mul(self); }
Tensor& square_(Tensor& self) { return self.mul_(self); }



Tensor& nan_to_num_out(const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf,
    Tensor& result) {
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "nan_to_num: dtype of out: ",
      result.scalar_type(),
      " should be same as input: ",
      self.scalar_type());

  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    at::native::resize_output(result, self.sizes());
    result.copy_(self);
    return result;
  }

  auto iter = TensorIterator::unary_op(result, self);
  nan_to_num_stub(iter.device_type(), iter, nan, pos_inf, neg_inf);
  return result;
}

Tensor nan_to_num(
    const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  auto result = at::empty_like(self);
  return at::nan_to_num_out(result, self, nan, pos_inf, neg_inf);
}

Tensor& nan_to_num_(
    Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  return at::nan_to_num_out(self, self, nan, pos_inf, neg_inf);
}

// Alias for trunc
Tensor& fix_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "fix is disabled in EasyFHE");
  return result;
}
Tensor fix(const Tensor& self) {
  TORCH_CHECK(false, "fix is disabled in EasyFHE");
  return self;
}
Tensor& fix_(Tensor& self) {
  TORCH_CHECK(false, "fix is disabled in EasyFHE");
  return self;
}

Tensor positive(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool, "The `+` operator, on a bool tensor is not supported.");
  return self;
}

Tensor& negative_out(const Tensor& self, Tensor& result) { return at::neg_out(result, self); }
Tensor negative(const Tensor& self) { return self.neg(); }
Tensor& negative_(Tensor& self) { return self.neg_(); }

Tensor logical_not(const Tensor& self) {
  TORCH_CHECK(false, "logical_not is disabled in EasyFHE");
  return self;
}

Tensor& logical_not_(Tensor& self) {
  TORCH_CHECK(false, "logical_not is disabled in EasyFHE");
  return self;
}

Tensor& logical_not_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(false, "logical_not is disabled in EasyFHE");
  return result;
}

std::tuple<Tensor, Tensor> frexp(const Tensor& self) {
  TORCH_CHECK(false, "frexp is disabled in EasyFHE");
  return std::tuple<Tensor, Tensor>(self, self);
}

std::tuple<Tensor&, Tensor&> frexp_out(const Tensor& self,
                                       Tensor& mantissa, Tensor& exponent) {
  TORCH_CHECK(false, "frexp is disabled in EasyFHE");
  return std::tuple<Tensor&, Tensor&>(mantissa, exponent);
}


DEFINE_DISPATCH(abs_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(angle_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(conj_physical_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(acos_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(acosh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(asinh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(atanh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(asin_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(atan_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bitwise_not_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(ceil_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cos_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cosh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(exp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(exp2_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(expm1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(floor_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(frac_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(frexp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log10_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log1p_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log2_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logical_not_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(neg_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(nan_to_num_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(reciprocal_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(round_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(round_decimals_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(rsqrt_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logit_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sign_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(signbit_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sgn_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sin_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sinh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sqrt_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(tan_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(trunc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(digamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(erfc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(erfinv_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(i0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(lgamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_airy_ai_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_j0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_j1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_y0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_y1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_entr_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_erfcx_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i0e_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i1e_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_log_ndtr_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_i0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_i1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_k0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_k1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_ndtri_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_scaled_modified_bessel_k0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_scaled_modified_bessel_k1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_spherical_bessel_j0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(trigamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(polygamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace at::native
