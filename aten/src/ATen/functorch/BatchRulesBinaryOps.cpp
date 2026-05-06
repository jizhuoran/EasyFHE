// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>

#include <utility>

namespace at::functorch {

template <typename F, F Func, typename... ExtraArgs>
static Tensor _binary_pointwise_batch_rule(
    const Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
    const Tensor& other, std::optional<int64_t> other_batch_dim,
    ExtraArgs... extra_args) {

  auto [tensor_, other_]= _binary_pointwise_helper(
      tensor, tensor_batch_dim, other, other_batch_dim);

  return Func(tensor_, std::move(other_), std::forward<ExtraArgs>(extra_args)...);
}

template <typename A, A a, typename C>
struct BinaryPointwiseBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename... T>
struct BinaryPointwiseBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  static std::tuple<Tensor, std::optional<int64_t>> apply(
      const Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
      const Tensor& other, std::optional<int64_t> other_batch_dim,
      T... extra_args) {
    return std::tuple(_binary_pointwise_batch_rule<F, Func, T...>(
        tensor, tensor_batch_dim, other, other_batch_dim,
        std::forward<T>(extra_args)...), 0);
  }
};

#define BINARY_POINTWISE_BATCH_RULE(fn) SINGLE_ARG(\
    BinaryPointwiseBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

template <typename A, A a, typename C>
struct BinaryRandomPointwiseBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename... T>
struct BinaryRandomPointwiseBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  static Tensor apply(const Tensor& tensor, const Tensor& other, T... extra_args) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
    auto maybe_layer = maybeCurrentDynamicLayer();
    TORCH_INTERNAL_ASSERT(maybe_layer.has_value())
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto cur_level = maybe_layer->layerId();
    RandomnessType randomness = maybe_layer->randomness();

    auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(tensor, cur_level);

    auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);

    check_randomness(randomness, (tensor_bdim || other_bdim));
    if (randomness == RandomnessType::Different && !tensor_bdim && !other_bdim) {
      auto shape = tensor_value.sizes();
      VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
      shapeVec.reserve(shape.size() + 1);
      shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());

      // not taken care of with binary batch rule, which assumes at least one input is batched
      tensor_value = tensor_value.expand_symint(shapeVec);
      tensor_bdim = 0;
    } else if (randomness == RandomnessType::Same && !tensor_bdim && !other_bdim) {

      // avoids unnecessary checks and batch rule assuming output is batched
      return Func(tensor_value, other_value, std::forward<T>(extra_args)...);
    }
    auto res = _binary_pointwise_batch_rule<F, Func, T...>(
      tensor_value, tensor_bdim, other_value, other_bdim,
      std::forward<T>(extra_args)...);
    return makeBatched(std::move(res), 0, cur_level);
  }
};

#define BINARY_RANDOM_POINTWISE_BATCH_RULE(fn) SINGLE_ARG(\
    BinaryRandomPointwiseBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

template <typename M, M Meth, typename... ExtraArgs>
static void binary_pointwise_inplace_batch_rule(
    Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
    const Tensor& other, std::optional<int64_t> other_batch_dim,
    ExtraArgs... extra_args) {
  if (!tensor_batch_dim && other_batch_dim) {
    vmapIncompatibleInplaceError("inplace arithmetic");
  }

  // compute max logical rank
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  (tensor_.*Meth)(other_, std::forward<ExtraArgs>(extra_args)...);
}

template <typename F, F Func>
static std::tuple<Tensor, std::optional<int64_t>> comparison_pointwise_batch_rule(
    const Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
    const Tensor& other, std::optional<int64_t> other_batch_dim) {
  // compute max logical rank
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  auto result = Func(tensor_, other_);
  return std::make_tuple( std::move(result), 0 );
}

static std::tuple<Tensor, std::optional<int64_t>> where_self_batch_rule(
    const Tensor& condition, std::optional<int64_t> condition_bdim,
    const Tensor& self, std::optional<int64_t> self_bdim, const Tensor& other, std::optional<int64_t> other_bdim) {
  auto condition_logical_rank = rankWithoutBatchDim(condition, condition_bdim);
  auto tensor_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  auto max_logical_rank = std::max({tensor_logical_rank, other_logical_rank, condition_logical_rank});

  auto condition_ = moveBatchDimToFront(condition, condition_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(other, other_bdim);

  condition_ = maybePadToLogicalRank(condition_, condition_bdim, max_logical_rank);
  self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_bdim, max_logical_rank);
  return std::make_tuple(at::where(condition_, self_, other_), 0);
}


static std::tuple<Tensor, std::optional<int64_t>> masked_select_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim,
    const Tensor& mask, std::optional<int64_t> mask_bdim) {
  TORCH_CHECK(!mask_bdim.has_value(),
      "vmap: Attempted to vmap over `mask` in torch.masked_select(self, mask) ",
      "We cannot support this because for each batch this would return a ",
      "differently shaped Tensor. "
      "Please voice your support in https://github.com/pytorch/functorch/issues/256");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  const auto batch_size = self_.size(0);
  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  const auto max_logical_rank = std::max(self_logical_rank, mask.dim());
  self_ = maybePadToLogicalRank(self_, 0, max_logical_rank);

  // masked_select returns a 1D tensor, so we have to reshape it into 2D
  auto result = at::masked_select(self_, mask).view({ batch_size, -1 });
  return std::make_tuple(std::move(result), 0);
}

static std::tuple<Tensor, std::optional<int64_t>> masked_select_backward_batch_rule(
    const Tensor& grad, std::optional<int64_t> grad_bdim,
    const Tensor& self, std::optional<int64_t> self_bdim,
    const Tensor& mask, std::optional<int64_t> mask_bdim) {
  TORCH_CHECK(!mask_bdim.has_value(),
      "vmap: Attempted to vmap over `mask` in torch.masked_select_backward(grad, self, mask) ",
      "We cannot support this because for each batch this would return a ",
      "differently shaped Tensor. "
      "Please voice your support in https://github.com/pytorch/functorch/issues/256");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto grad_ = moveBatchDimToFront(grad, grad_bdim);

  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  const auto max_logical_rank = std::max(self_logical_rank, mask.dim());

  self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);

  const auto batch_size = get_bdim_size2(grad, grad_bdim, self, self_bdim);
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), batch_size);

  auto result = at::masked_select_backward(grad_, self_.contiguous(), mask);
  return std::make_tuple(std::move(result), 0);
}

static void fill__Tensor_batch_rule(
    Tensor& self,
    std::optional<int64_t> self_bdim,
    const Tensor& other,
    std::optional<int64_t> other_bdim) {
  if (!other_bdim.has_value()) {
    // Optimization: fill_ is faster than the other path which does
    // reshaping + copy_
    self.fill_(other);
    return;
  }
  if (!self_bdim) {
    vmapIncompatibleInplaceError("fill_");
  }
  auto self_and_other = _binary_pointwise_helper(
      self, self_bdim, other, other_bdim, /*do_type_promotion*/false);
  std::get<0>(self_and_other).copy_(std::get<1>(self_and_other));
}

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  #define BINARY_RANDOM_POINTWISE(op) \
    m.impl(#op, BINARY_RANDOM_POINTWISE_BATCH_RULE(ATEN_FN(op)));
  #define BINARY_RANDOM_POINTWISE2(op, overload) \
    m.impl(#op"."#overload, BINARY_RANDOM_POINTWISE_BATCH_RULE(ATEN_FN2(op, overload)));

  BINARY_RANDOM_POINTWISE2(normal, Tensor_Tensor);
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
#define BINARY_POINTWISE2(op, overload) \
  VMAP_SUPPORT2(op, overload, BINARY_POINTWISE_BATCH_RULE(ATEN_FN2(op, overload)));
#define BINARY_POINTWISE(op) \
  VMAP_SUPPORT(op, BINARY_POINTWISE_BATCH_RULE(ATEN_FN(op)));
#define UNARY_POINTWISE2(op, overload) \
  VMAP_SUPPORT2(op, overload, BASIC_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));
#define UNARY_POINTWISE(op) \
  VMAP_SUPPORT(op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));
#define UNARY_SCALAR_POINTWISE2(op, overload) \
  VMAP_SUPPORT(op, overload, SCALAR_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));

#define BINARY_SCALAR_2(op, tensor_tensor, tensor_scalar) \
  BINARY_POINTWISE2(op, tensor_tensor);\
  UNARY_POINTWISE2(op, tensor_scalar);

// For all 3 combinations of Tensor x Tensor, Tensor x Scalar, Scalar x Tensor
#define BINARY_SCALAR_3(op, tensor_tensor, tensor_scalar, scalar_tensor) \
  BINARY_POINTWISE2(op, tensor_tensor);\
  UNARY_POINTWISE2(op, tensor_scalar);\
  POINTWISE_BOXED(op.scalar_tensor);

#define BINARY_SCALAR_3_Tensor(op, tensor_scalar, scalar_tensor) \
  BINARY_POINTWISE(op);\
  UNARY_POINTWISE2(op, tensor_scalar);\
  POINTWISE_BOXED(op.scalar_tensor);

  // Batching rule registrations start
  POINTWISE_BOXED(__ilshift__.Tensor);
  POINTWISE_BOXED(__ilshift__.Scalar);
  POINTWISE_BOXED(__irshift__.Tensor)
  POINTWISE_BOXED(__irshift__.Scalar)
  BINARY_SCALAR_2(__lshift__, Tensor, Scalar);
  BINARY_SCALAR_2(__rshift__, Tensor, Scalar);

  BINARY_SCALAR_2(add, Tensor, Scalar);
  POINTWISE_BOXED(addcdiv);
  POINTWISE_BOXED(addcmul);
  BINARY_POINTWISE(atan2);
  BINARY_SCALAR_2(bitwise_and, Tensor, Scalar);
  BINARY_POINTWISE2(bitwise_and_, Tensor);
  POINTWISE_BOXED(bitwise_and_.Scalar);
  POINTWISE_BOXED(bitwise_and.Scalar_Tensor);
  BINARY_SCALAR_2(bitwise_or, Tensor, Scalar);
  BINARY_POINTWISE2(bitwise_or_, Tensor);
  POINTWISE_BOXED(bitwise_or_.Scalar);
  POINTWISE_BOXED(bitwise_or.Scalar_Tensor);
  BINARY_SCALAR_2(bitwise_xor, Tensor, Scalar);
  BINARY_POINTWISE2(bitwise_xor_, Tensor);
  POINTWISE_BOXED(bitwise_xor_.Scalar);
  POINTWISE_BOXED(bitwise_xor.Scalar_Tensor);
  BINARY_SCALAR_3(bitwise_left_shift, Tensor, Tensor_Scalar, Scalar_Tensor);
  POINTWISE_BOXED(bitwise_left_shift_.Tensor_Scalar);
  POINTWISE_BOXED(bitwise_left_shift_.Tensor);
  BINARY_SCALAR_3(bitwise_right_shift, Tensor, Tensor_Scalar, Scalar_Tensor);
  POINTWISE_BOXED(bitwise_right_shift_.Tensor_Scalar);
  POINTWISE_BOXED(bitwise_right_shift_.Tensor);

  UNARY_POINTWISE(clamp);
  POINTWISE_BOXED(clamp.Tensor);
  BINARY_POINTWISE2(clamp_min, Tensor);
  UNARY_POINTWISE(clamp_min);
  POINTWISE_BOXED(clamp_min_);
  BINARY_POINTWISE2(clamp_max, Tensor);
  UNARY_POINTWISE(clamp_max);
  POINTWISE_BOXED(clamp_max_);
  BINARY_POINTWISE(complex);

  BINARY_SCALAR_2(copysign, Tensor, Scalar);
  POINTWISE_BOXED(copysign_.Tensor);
  POINTWISE_BOXED(copysign_.Scalar);
  BINARY_SCALAR_2(div, Tensor, Scalar);
  BINARY_SCALAR_2(div, Tensor_mode, Scalar_mode);

  BINARY_POINTWISE(floor_divide);
  UNARY_POINTWISE2(floor_divide, Scalar);

  BINARY_POINTWISE(fmax);
  BINARY_POINTWISE(fmin);
  BINARY_SCALAR_2(fmod, Tensor, Scalar);
  POINTWISE_BOXED(frexp.Tensor);
  BINARY_POINTWISE(heaviside);
  BINARY_POINTWISE(hypot);
  BINARY_POINTWISE(gcd);
  BINARY_POINTWISE2(ldexp, Tensor);
  BINARY_POINTWISE(logaddexp);
  BINARY_POINTWISE(logaddexp2);
  POINTWISE_BOXED(lerp.Scalar);
  POINTWISE_BOXED(lerp.Tensor);
  BINARY_POINTWISE(lcm);
  BINARY_POINTWISE(maximum);
  BINARY_POINTWISE(minimum);

  BINARY_SCALAR_2(mul, Tensor, Scalar);
  BINARY_POINTWISE(nextafter);
  BINARY_SCALAR_3(pow, Tensor_Tensor, Tensor_Scalar, Scalar);
  POINTWISE_BOXED2(pow_, Scalar);
  BINARY_POINTWISE(polar);
  BINARY_SCALAR_2(sub, Tensor, Scalar);
  BINARY_SCALAR_3(remainder, Tensor, Scalar, Scalar_Tensor);
  BINARY_SCALAR_2(rsub, Tensor, Scalar);

  VMAP_SUPPORT2(where, self, where_self_batch_rule);

  BINARY_POINTWISE(sigmoid_backward);
  BINARY_POINTWISE(tanh_backward);

  using TensorScalarInplaceT = Tensor& (Tensor::*)(const Tensor&, const Scalar&) const;
  using ScalarScalarInplaceT = Tensor& (Tensor::*)(const Scalar&, const Scalar&) const;
  using TensorInplaceT = Tensor& (Tensor::*)(const Tensor&) const;
  using TensorInplaceModeT = Tensor& (Tensor::*)(const Tensor&, std::optional<std::string_view>) const;
  using ScalarInplaceT = Tensor& (Tensor::*)(const Scalar&) const;
  using CopyT = Tensor& (Tensor::*)(const Tensor&, bool) const;

  POINTWISE_BOXED(add_.Tensor); // just testing
  POINTWISE_BOXED(atan2_);
  POINTWISE_BOXED(gcd_);
  POINTWISE_BOXED(lcm_);
  VMAP_SUPPORT2(add_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarScalarInplaceT, &Tensor::add_, const Scalar&, const Scalar&>));
  VMAP_SUPPORT2(sub_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorScalarInplaceT, &Tensor::sub_, const Scalar&>));
  VMAP_SUPPORT2(sub_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarScalarInplaceT, &Tensor::sub_, const Scalar&, const Scalar&>));
  VMAP_SUPPORT2(mul_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::mul_>));
  VMAP_SUPPORT2(mul_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarInplaceT, &Tensor::mul_, const Scalar&>));
  VMAP_SUPPORT2(div_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::div_>));
  VMAP_SUPPORT2(div_, Tensor_mode, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceModeT, &Tensor::div_, std::optional<std::string_view>>));
  VMAP_SUPPORT2(div_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarInplaceT, &Tensor::div_, const Scalar&>));
  VMAP_SUPPORT2(clamp_min_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::clamp_min_>));
  VMAP_SUPPORT2(clamp_max_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::clamp_max_>));
  VMAP_SUPPORT2(masked_fill_, Scalar, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorScalarInplaceT, &Tensor::masked_fill_, const Scalar&>));
  VMAP_SUPPORT(copy_, SINGLE_ARG(binary_pointwise_inplace_batch_rule<CopyT, &Tensor::copy_, bool>));

#define COMPARISON_POINTWISE(op) \
  VMAP_SUPPORT2(op, Tensor, \
      SINGLE_ARG(comparison_pointwise_batch_rule<decltype(&ATEN_FN2(op, Tensor)), &at::op>)); \
  UNARY_POINTWISE2(op, Scalar)

  COMPARISON_POINTWISE(eq);
  COMPARISON_POINTWISE(gt);
  COMPARISON_POINTWISE(ge);
  COMPARISON_POINTWISE(le);
  COMPARISON_POINTWISE(lt);
  COMPARISON_POINTWISE(ne);

#undef COMPARISON_POINTWISE
#undef BINARY_POINTWISE2
#undef BINARY_POINTWISE
#undef UNARY_POINTWISE2
#undef UNARY_POINTWISE
#undef UNARY_SCALAR_POINTWISE2
#undef BINARY_SCALAR_3

#define LOGICAL_COMPARISON_POINTWISE(op) \
  VMAP_SUPPORT(op, \
      SINGLE_ARG(comparison_pointwise_batch_rule<decltype(&ATEN_FN(op)), &ATEN_FN(op)>)); \
  VMAP_SUPPORT(op ## _, \
      SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor:: op ## _ >));

  LOGICAL_COMPARISON_POINTWISE(logical_and);
  LOGICAL_COMPARISON_POINTWISE(logical_or);
  LOGICAL_COMPARISON_POINTWISE(logical_xor);

#undef SINGLE_ARG
#undef LOGICAL_COMPARISON_POINTWISE
  VMAP_SUPPORT(masked_select, masked_select_batch_rule);
  VMAP_SUPPORT(masked_select_backward, masked_select_backward_batch_rule);

  VMAP_SUPPORT2(fill_, Tensor, fill__Tensor_batch_rule);
}

} // namespace at::functorch
