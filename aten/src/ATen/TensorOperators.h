#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <string_view>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

namespace at {

#define AT_FORALL_BINARY_OPS(_)                                             \
  _(+, x.add(y), y.add(x))                                                  \
  _(*, x.mul(y), y.mul(x))                                                  \
  _(-,                                                                      \
    x.sub(y),                                                               \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).sub_(y))       \
  _(/,                                                                      \
    x.div(y),                                                               \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).div_(y))       \
  _(%,                                                                      \
    x.remainder(y),                                                         \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).remainder_(y))

#define DEFINE_OPERATOR(op, body, reverse_scalar_body)          \
  inline Tensor operator op(const Tensor& x, const Tensor& y) { \
    return body;                                                \
  }                                                             \
  inline Tensor operator op(const Tensor& x, const Scalar& y) { \
    return body;                                                \
  }                                                             \
  inline Tensor operator op(const Scalar& x, const Tensor& y) { \
    return reverse_scalar_body;                                 \
  }

AT_FORALL_BINARY_OPS(DEFINE_OPERATOR)
#undef DEFINE_OPERATOR
#undef AT_FORALL_BINARY_OPS

inline Tensor _easyfhe_disabled_tensor_op(std::string_view name) {
  TORCH_CHECK(false, name, " is disabled in EasyFHE");
  return Tensor();
}

#define DEFINE_DISABLED_COMPARE_OPERATOR(op, name)                  \
  inline Tensor operator op(const Tensor& x, const Tensor& y) {     \
    return _easyfhe_disabled_tensor_op(name);                       \
  }                                                                 \
  inline Tensor operator op(const Tensor& x, const Scalar& y) {     \
    return _easyfhe_disabled_tensor_op(name);                       \
  }                                                                 \
  inline Tensor operator op(const Scalar& x, const Tensor& y) {     \
    return _easyfhe_disabled_tensor_op(name);                       \
  }

DEFINE_DISABLED_COMPARE_OPERATOR(==, "eq")
DEFINE_DISABLED_COMPARE_OPERATOR(!=, "ne")
DEFINE_DISABLED_COMPARE_OPERATOR(<, "lt")
DEFINE_DISABLED_COMPARE_OPERATOR(<=, "le")
DEFINE_DISABLED_COMPARE_OPERATOR(>, "gt")
DEFINE_DISABLED_COMPARE_OPERATOR(>=, "ge")

#undef DEFINE_DISABLED_COMPARE_OPERATOR

inline Tensor logical_and(const Tensor& self, const Tensor& other) {
  return _easyfhe_disabled_tensor_op("logical_and");
}

inline Tensor logical_or(const Tensor& self, const Tensor& other) {
  return _easyfhe_disabled_tensor_op("logical_or");
}

inline Tensor bitwise_and(const Tensor& self, const Tensor& other) {
  return _easyfhe_disabled_tensor_op("bitwise_and");
}

inline Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim=false) {
  return _easyfhe_disabled_tensor_op("amax");
}

inline Tensor amin(const Tensor& self, IntArrayRef dim, bool keepdim=false) {
  return _easyfhe_disabled_tensor_op("amin");
}

inline Tensor sum(const Tensor& self) {
  return _easyfhe_disabled_tensor_op("sum");
}

inline std::tuple<Tensor, Tensor> max(
    const Tensor& self,
    int64_t dim,
    bool keepdim=false) {
  TORCH_CHECK(false, "max is disabled in EasyFHE");
  return std::tuple<Tensor, Tensor>(Tensor(), Tensor());
}

inline Tensor scatter_reduce(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    std::string_view reduce,
    bool include_self) {
  return _easyfhe_disabled_tensor_op("scatter_reduce");
}

} // namespace at
