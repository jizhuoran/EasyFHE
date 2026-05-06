#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/util/typeid.h>
#include <cstdint>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>

#include <ATen/BlasBackend.h>
#include <ATen/ceil_div.h>

#ifdef USE_MSLK
#include <mslk/gemm/gemm_torch.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/abs.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones.h>
#endif

using at::blas::ScalingType;

namespace at::native::onednn::scaled {

/**
 * Track concrete implementations available
 */
enum class ScaledGemmImplementation {
  NONE = 0,
  TENSORWISE_TENSORWISE = 1,
  ROWWISE_ROWWISE = 2,
};

/**
 * Convert passed int (enum) from python back into a
 * strictly-typed enum
 */
template <class EnumType, class ArrayType>
std::vector<EnumType> convert_int_to_enum(ArrayType& v) {
  std::vector<EnumType> converted;
  converted.reserve(v.size());

  for (auto vi : v) {
    converted.push_back(static_cast<EnumType>(vi));
  }
  return converted;
}

bool check_tensorwise_recipe(
    c10::ScalarType,
    std::vector<ScalingType>&,
    ArrayRef<Tensor>&,
    c10::ScalarType,
    std::vector<ScalingType>&,
    ArrayRef<Tensor>&);

bool check_rowwise_recipe(
    c10::ScalarType,
    std::vector<ScalingType>&,
    ArrayRef<Tensor>&,
    c10::ScalarType,
    std::vector<ScalingType>&,
    ArrayRef<Tensor>&);

} // namespace at::native::onednn::scaled
