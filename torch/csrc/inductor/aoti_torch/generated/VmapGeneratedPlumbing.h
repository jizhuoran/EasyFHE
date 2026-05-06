
#pragma once
#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at { namespace functorch {

template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Byte_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Byte::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Char_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Char::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Double_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Double::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Float_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Float::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Int_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Int::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Long_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Long::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Short_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Short::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cast_Half_generated_plumbing(const at::Tensor & self, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_cast_Half::call(self, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _backward_generated_plumbing(const at::Tensor & self, at::TensorList inputs, const ::std::optional<at::Tensor> & gradient, ::std::optional<bool> retain_graph, bool create_graph) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(inputs, cur_level) && !isBatchedAtLevel(gradient, cur_level)) {
    return at::_ops::_backward::call(self, inputs, gradient, retain_graph, create_graph);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> gradient_value;
  std::optional<int64_t> gradient_bdim;
  if (gradient) {
      std::tie(gradient_value, gradient_bdim) = unwrapTensorAtLevel(gradient.value(), cur_level);
  }
  batch_rule(self_value, self_bdim, inputs, gradient_value, gradient_bdim, retain_graph, create_graph);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void set_data_generated_plumbing(at::Tensor & self, const at::Tensor & new_data) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(new_data, cur_level)) {
    return at::_ops::set_data::call(self, new_data);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [new_data_value, new_data_bdim] = unwrapTensorAtLevel(new_data, cur_level);
  batch_rule(self_value, self_bdim, new_data_value, new_data_bdim);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor data_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::data::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & requires_grad__generated_plumbing(at::Tensor & self, bool requires_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::requires_grad_::call(self, requires_grad);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, requires_grad);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void retain_grad_generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::retain_grad::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _fw_primal_generated_plumbing(const at::Tensor & self, int64_t level) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_fw_primal::call(self, level);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, level);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _make_dual_generated_plumbing(const at::Tensor & primal, const at::Tensor & tangent, int64_t level) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(primal, cur_level) && !isBatchedAtLevel(tangent, cur_level)) {
    return at::_ops::_make_dual::call(primal, tangent, level);
  }
  auto [primal_value, primal_bdim] = unwrapTensorAtLevel(primal, cur_level);
  auto [tangent_value, tangent_bdim] = unwrapTensorAtLevel(tangent, cur_level);
  auto results = batch_rule(primal_value, primal_bdim, tangent_value, tangent_bdim, level);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> _unpack_dual_generated_plumbing(const at::Tensor & dual, int64_t level) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(dual, cur_level)) {
    return at::_ops::_unpack_dual::call(dual, level);
  }
  auto [dual_value, dual_bdim] = unwrapTensorAtLevel(dual, cur_level);
  auto results = batch_rule(dual_value, dual_bdim, level);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _new_zeros_with_same_feature_meta_generated_plumbing(const at::Tensor & self, const at::Tensor & other, int64_t self_num_batch_dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::_new_zeros_with_same_feature_meta::call(self, other, self_num_batch_dims);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, self_num_batch_dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rename_generated_plumbing(const at::Tensor & self, ::std::optional<at::DimnameList> names) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rename::call(self, names);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor align_to_generated_plumbing(const at::Tensor & self, at::DimnameList names) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::align_to::call(self, names);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor align_to_ellipsis_idx_generated_plumbing(const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::align_to_ellipsis_idx::call(self, order, ellipsis_idx);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, order, ellipsis_idx);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor align_as_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::align_as::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> align_tensors_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::align_tensors::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _assert_async_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_assert_async::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _assert_async_msg_generated_plumbing(const at::Tensor & self, c10::string_view assert_msg) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_assert_async_msg::call(self, assert_msg);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, assert_msg);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _functional_assert_scalar_generated_plumbing(const at::Scalar & self, c10::string_view assert_msg, const at::Tensor & dep_token) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(dep_token, cur_level)) {
    return at::_ops::_functional_assert_scalar::call(self, assert_msg, dep_token);
  }
  auto [dep_token_value, dep_token_bdim] = unwrapTensorAtLevel(dep_token, cur_level);
  auto results = batch_rule(self, assert_msg, dep_token_value, dep_token_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _functional_assert_async_msg_generated_plumbing(const at::Tensor & self, c10::string_view assert_msg, const at::Tensor & dep_token) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(dep_token, cur_level)) {
    return at::_ops::_functional_assert_async_msg::call(self, assert_msg, dep_token);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [dep_token_value, dep_token_bdim] = unwrapTensorAtLevel(dep_token, cur_level);
  auto results = batch_rule(self_value, self_bdim, assert_msg, dep_token_value, dep_token_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _assert_tensor_metadata_generated_plumbing(const at::Tensor & a, at::OptionalSymIntArrayRef size, at::OptionalSymIntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Device> device, ::std::optional<at::Layout> layout) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(a, cur_level)) {
    return at::_ops::_assert_tensor_metadata::call(a, size, stride, dtype, device, layout);
  }
  auto [a_value, a_bdim] = unwrapTensorAtLevel(a, cur_level);
  batch_rule(a_value, a_bdim, size, stride, dtype, device, layout);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _functional_sym_constrain_range_generated_plumbing(const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max, const at::Tensor & dep_token) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(dep_token, cur_level)) {
    return at::_ops::_functional_sym_constrain_range::call(size, min, max, dep_token);
  }
  auto [dep_token_value, dep_token_bdim] = unwrapTensorAtLevel(dep_token, cur_level);
  auto results = batch_rule(size, min, max, dep_token_value, dep_token_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _functional_sym_constrain_range_for_size_generated_plumbing(const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max, const at::Tensor & dep_token) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(dep_token, cur_level)) {
    return at::_ops::_functional_sym_constrain_range_for_size::call(size, min, max, dep_token);
  }
  auto [dep_token_value, dep_token_bdim] = unwrapTensorAtLevel(dep_token, cur_level);
  auto results = batch_rule(size, min, max, dep_token_value, dep_token_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor refine_names_generated_plumbing(const at::Tensor & self, at::DimnameList names) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::refine_names::call(self, names);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _reshape_from_tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & shape) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(shape, cur_level)) {
    return at::_ops::_reshape_from_tensor::call(self, shape);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [shape_value, shape_bdim] = unwrapTensorAtLevel(shape, cur_level);
  auto results = batch_rule(self_value, self_bdim, shape_value, shape_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _shape_as_tensor_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_shape_as_tensor::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor abs_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::abs::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & abs__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::abs_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor absolute_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::absolute::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & absolute__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::absolute_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor angle_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::angle::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_as_real_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_as_real::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_as_complex_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_as_complex::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sgn_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sgn::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sgn__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sgn_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor real_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::real::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor imag_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::imag::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _conj_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_conj::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor conj_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::conj::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _conj_physical_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_conj_physical::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor conj_physical_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::conj_physical::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & conj_physical__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::conj_physical_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor resolve_conj_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::resolve_conj::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor resolve_neg_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::resolve_neg::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _neg_view_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_neg_view::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor acos_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::acos::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & acos__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::acos_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor arccos_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arccos::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & arccos__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arccos_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor add_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::add_Tensor::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & add__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::add__Tensor::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor add_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::add_Scalar::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & add__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::add__Scalar::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _is_all_true_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_is_all_true::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _is_any_true_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_is_any_true::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor all_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::all_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor all_dims_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::all_dims::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor all_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::all_dimname::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor any_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::any_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor any_dims_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::any_dims::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor any_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::any_dimname::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _dim_arange_generated_plumbing(const at::Tensor & like, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(like, cur_level)) {
    return at::_ops::_dim_arange::call(like, dim);
  }
  auto [like_value, like_bdim] = unwrapTensorAtLevel(like, cur_level);
  auto results = batch_rule(like_value, like_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor argmax_generated_plumbing(const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::argmax::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor argmin_generated_plumbing(const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::argmin::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor acosh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::acosh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & acosh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::acosh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor arccosh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arccosh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & arccosh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arccosh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor asinh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::asinh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & asinh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::asinh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor arcsinh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arcsinh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & arcsinh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arcsinh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atanh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atanh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & atanh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atanh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor arctanh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arctanh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & arctanh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arctanh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor as_strided_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::as_strided::call(self, size, stride, storage_offset);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride, storage_offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor asin_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::asin::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & asin__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::asin_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor arcsin_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arcsin::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & arcsin__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arcsin_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atan_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atan::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & atan__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atan_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor arctan_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arctan::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & arctan__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::arctan_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atleast_1d_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atleast_1d::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> atleast_1d_Sequence_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::atleast_1d_Sequence::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atleast_2d_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atleast_2d::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> atleast_2d_Sequence_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::atleast_2d_Sequence::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atleast_3d_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atleast_3d::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> atleast_3d_Sequence_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::atleast_3d_Sequence::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bernoulli_generated_plumbing(const at::Tensor & self, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bernoulli::call(self, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bernoulli__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(p, cur_level)) {
    return at::_ops::bernoulli__Tensor::call(self, p, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [p_value, p_bdim] = unwrapTensorAtLevel(p, cur_level);
  batch_rule(self_value, self_bdim, p_value, p_bdim, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bernoulli__float_generated_plumbing(at::Tensor & self, double p, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bernoulli__float::call(self, p, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, p, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bernoulli_p_generated_plumbing(const at::Tensor & self, double p, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bernoulli_p::call(self, p, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_not_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_not::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_not__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_not_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor copysign_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::copysign_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & copysign__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::copysign__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor copysign_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::copysign_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & copysign__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::copysign__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _lazy_clone_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_lazy_clone::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_not_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logical_not::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & logical_not__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logical_not_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_xor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_xor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & logical_xor__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_xor_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_and_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_and::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & logical_and__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_and_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_or_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_or::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & logical_or__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_or_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> broadcast_tensors_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::broadcast_tensors::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor broadcast_to_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::broadcast_to::call(self, size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cat_generated_plumbing(const at::ITensorListRef & tensors, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::cat::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cat_names_generated_plumbing(at::TensorList tensors, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::cat_names::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor concat_generated_plumbing(at::TensorList tensors, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::concat::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor concat_names_generated_plumbing(at::TensorList tensors, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::concat_names::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor concatenate_generated_plumbing(at::TensorList tensors, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::concatenate::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor concatenate_names_generated_plumbing(at::TensorList tensors, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::concatenate_names::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor block_diag_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::block_diag::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ceil_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ceil::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & ceil__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ceil_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> unsafe_chunk_generated_plumbing(const at::Tensor & self, int64_t chunks, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unsafe_chunk::call(self, chunks, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, chunks, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> chunk_generated_plumbing(const at::Tensor & self, int64_t chunks, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::chunk::call(self, chunks, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, chunks, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> tensor_split_sections_generated_plumbing(const at::Tensor & self, c10::SymInt sections, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tensor_split_sections::call(self, sections, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sections, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> tensor_split_indices_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tensor_split_indices::call(self, indices, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> tensor_split_tensor_indices_or_sections_generated_plumbing(const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(tensor_indices_or_sections, cur_level)) {
    return at::_ops::tensor_split_tensor_indices_or_sections::call(self, tensor_indices_or_sections, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [tensor_indices_or_sections_value, tensor_indices_or_sections_bdim] = unwrapTensorAtLevel(tensor_indices_or_sections, cur_level);
  auto results = batch_rule(self_value, self_bdim, tensor_indices_or_sections_value, tensor_indices_or_sections_bdim, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_Tensor_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clamp_Tensor::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> min_value;
  std::optional<int64_t> min_bdim;
  if (min) {
      std::tie(min_value, min_bdim) = unwrapTensorAtLevel(min.value(), cur_level);
  }
  std::optional<Tensor> max_value;
  std::optional<int64_t> max_bdim;
  if (max) {
      std::tie(max_value, max_bdim) = unwrapTensorAtLevel(max.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, min_value, min_bdim, max_value, max_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clamp__generated_plumbing(at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp_::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, min, max);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clamp__Tensor_generated_plumbing(at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clamp__Tensor::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> min_value;
  std::optional<int64_t> min_bdim;
  if (min) {
      std::tie(min_value, min_bdim) = unwrapTensorAtLevel(min.value(), cur_level);
  }
  std::optional<Tensor> max_value;
  std::optional<int64_t> max_bdim;
  if (max) {
      std::tie(max_value, max_bdim) = unwrapTensorAtLevel(max.value(), cur_level);
  }
  batch_rule(self_value, self_bdim, min_value, min_bdim, max_value, max_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_max_generated_plumbing(const at::Tensor & self, const at::Scalar & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp_max::call(self, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_max_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clamp_max_Tensor::call(self, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [max_value, max_bdim] = unwrapTensorAtLevel(max, cur_level);
  auto results = batch_rule(self_value, self_bdim, max_value, max_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clamp_max__generated_plumbing(at::Tensor & self, const at::Scalar & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp_max_::call(self, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, max);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clamp_max__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clamp_max__Tensor::call(self, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [max_value, max_bdim] = unwrapTensorAtLevel(max, cur_level);
  batch_rule(self_value, self_bdim, max_value, max_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_min_generated_plumbing(const at::Tensor & self, const at::Scalar & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp_min::call(self, min);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_min_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level)) {
    return at::_ops::clamp_min_Tensor::call(self, min);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [min_value, min_bdim] = unwrapTensorAtLevel(min, cur_level);
  auto results = batch_rule(self_value, self_bdim, min_value, min_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clamp_min__generated_plumbing(at::Tensor & self, const at::Scalar & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp_min_::call(self, min);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, min);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clamp_min__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level)) {
    return at::_ops::clamp_min__Tensor::call(self, min);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [min_value, min_bdim] = unwrapTensorAtLevel(min, cur_level);
  batch_rule(self_value, self_bdim, min_value, min_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clip_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clip::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clip_Tensor_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clip_Tensor::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> min_value;
  std::optional<int64_t> min_bdim;
  if (min) {
      std::tie(min_value, min_bdim) = unwrapTensorAtLevel(min.value(), cur_level);
  }
  std::optional<Tensor> max_value;
  std::optional<int64_t> max_bdim;
  if (max) {
      std::tie(max_value, max_bdim) = unwrapTensorAtLevel(max.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, min_value, min_bdim, max_value, max_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clip__generated_plumbing(at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clip_::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, min, max);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & clip__Tensor_generated_plumbing(at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clip__Tensor::call(self, min, max);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> min_value;
  std::optional<int64_t> min_bdim;
  if (min) {
      std::tie(min_value, min_bdim) = unwrapTensorAtLevel(min.value(), cur_level);
  }
  std::optional<Tensor> max_value;
  std::optional<int64_t> max_bdim;
  if (max) {
      std::tie(max_value, max_bdim) = unwrapTensorAtLevel(max.value(), cur_level);
  }
  batch_rule(self_value, self_bdim, min_value, min_bdim, max_value, max_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor complex_generated_plumbing(const at::Tensor & real, const at::Tensor & imag) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(real, cur_level) && !isBatchedAtLevel(imag, cur_level)) {
    return at::_ops::complex::call(real, imag);
  }
  auto [real_value, real_bdim] = unwrapTensorAtLevel(real, cur_level);
  auto [imag_value, imag_bdim] = unwrapTensorAtLevel(imag, cur_level);
  auto results = batch_rule(real_value, real_bdim, imag_value, imag_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor polar_generated_plumbing(const at::Tensor & abs, const at::Tensor & angle) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(abs, cur_level) && !isBatchedAtLevel(angle, cur_level)) {
    return at::_ops::polar::call(abs, angle);
  }
  auto [abs_value, abs_bdim] = unwrapTensorAtLevel(abs, cur_level);
  auto [angle_value, angle_bdim] = unwrapTensorAtLevel(angle, cur_level);
  auto results = batch_rule(abs_value, abs_bdim, angle_value, angle_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor contiguous_generated_plumbing(const at::Tensor & self, at::MemoryFormat memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::contiguous::call(self, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor copy_generated_plumbing(const at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::copy::call(self, src, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & copy__generated_plumbing(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::copy_::call(self, src, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  batch_rule(self_value, self_bdim, src_value, src_bdim, non_blocking);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _copy_from_generated_plumbing(const at::Tensor & self, const at::Tensor & dst, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(dst, cur_level)) {
    return at::_ops::_copy_from::call(self, dst, non_blocking);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [dst_value, dst_bdim] = unwrapTensorAtLevel(dst, cur_level);
  auto results = batch_rule(self_value, self_bdim, dst_value, dst_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _copy_from_and_resize_generated_plumbing(const at::Tensor & self, const at::Tensor & dst) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(dst, cur_level)) {
    return at::_ops::_copy_from_and_resize::call(self, dst);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [dst_value, dst_bdim] = unwrapTensorAtLevel(dst, cur_level);
  auto results = batch_rule(self_value, self_bdim, dst_value, dst_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cos_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cos::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & cos__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cos_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cosh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cosh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & cosh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cosh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor count_nonzero_dim_IntList_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::count_nonzero_dim_IntList::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor count_nonzero_generated_plumbing(const at::Tensor & self, ::std::optional<int64_t> dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::count_nonzero::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> cummax_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cummax::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> cummax_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cummax_dimname::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _cummax_helper_generated_plumbing(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(values, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::_cummax_helper::call(self, values, indices, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto [indices_value, indices_bdim] = unwrapTensorAtLevel(indices, cur_level);
  batch_rule(self_value, self_bdim, values_value, values_bdim, indices_value, indices_bdim, dim);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> cummin_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cummin::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> cummin_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cummin_dimname::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _cummin_helper_generated_plumbing(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(values, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::_cummin_helper::call(self, values, indices, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto [indices_value, indices_bdim] = unwrapTensorAtLevel(indices, cur_level);
  batch_rule(self_value, self_bdim, values_value, values_bdim, indices_value, indices_bdim, dim);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cummaxmin_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & indices, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::cummaxmin_backward::call(grad, input, indices, dim);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [indices_value, indices_bdim] = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim, indices_value, indices_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cumprod_generated_plumbing(const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumprod::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & cumprod__generated_plumbing(at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumprod_::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, dim, dtype);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cumprod_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumprod_dimname::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & cumprod__dimname_generated_plumbing(at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumprod__dimname::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, dim, dtype);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cumprod_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & input, int64_t dim, const at::Tensor & output) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::cumprod_backward::call(grad, input, dim, output);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [output_value, output_bdim] = unwrapTensorAtLevel(output, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim, dim, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cumsum_generated_plumbing(const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumsum::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & cumsum__generated_plumbing(at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumsum_::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, dim, dtype);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cumsum_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumsum_dimname::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & cumsum__dimname_generated_plumbing(at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cumsum__dimname::call(self, dim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, dim, dtype);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diag_embed_generated_plumbing(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diag_embed::call(self, offset, dim1, dim2);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagflat_generated_plumbing(const at::Tensor & self, int64_t offset) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diagflat::call(self, offset);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_generated_plumbing(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diagonal::call(self, offset, dim1, dim2);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_Dimname_generated_plumbing(const at::Tensor & self, at::Dimname outdim, at::Dimname dim1, at::Dimname dim2, int64_t offset) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diagonal_Dimname::call(self, outdim, dim1, dim2, offset);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, outdim, dim1, dim2, offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_backward_generated_plumbing(const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::diagonal_backward::call(grad_output, input_sizes, offset, dim1, dim2);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & fill_diagonal__generated_plumbing(at::Tensor & self, const at::Scalar & fill_value, bool wrap) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fill_diagonal_::call(self, fill_value, wrap);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, fill_value, wrap);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diff_generated_plumbing(const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(prepend, cur_level) && !isBatchedAtLevel(append, cur_level)) {
    return at::_ops::diff::call(self, n, dim, prepend, append);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> prepend_value;
  std::optional<int64_t> prepend_bdim;
  if (prepend) {
      std::tie(prepend_value, prepend_bdim) = unwrapTensorAtLevel(prepend.value(), cur_level);
  }
  std::optional<Tensor> append_value;
  std::optional<int64_t> append_bdim;
  if (append) {
      std::tie(append_value, append_bdim) = unwrapTensorAtLevel(append.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, n, dim, prepend_value, prepend_bdim, append_value, append_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> gradient_scalarint_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & spacing, ::std::optional<int64_t> dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gradient_scalarint::call(self, spacing, dim, edge_order);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> gradient_scalararray_generated_plumbing(const at::Tensor & self, const at::Scalar & spacing, at::IntArrayRef dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gradient_scalararray::call(self, spacing, dim, edge_order);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> gradient_array_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gradient_array::call(self, dim, edge_order);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> gradient_scalarrayint_generated_plumbing(const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, ::std::optional<int64_t> dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gradient_scalarrayint::call(self, spacing, dim, edge_order);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> gradient_scalarrayarray_generated_plumbing(const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, at::IntArrayRef dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gradient_scalarrayarray::call(self, spacing, dim, edge_order);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> gradient_tensorarrayint_generated_plumbing(const at::Tensor & self, at::TensorList spacing, ::std::optional<int64_t> dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(spacing, cur_level)) {
    return at::_ops::gradient_tensorarrayint::call(self, spacing, dim, edge_order);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> gradient_tensorarray_generated_plumbing(const at::Tensor & self, at::TensorList spacing, at::IntArrayRef dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(spacing, cur_level)) {
    return at::_ops::gradient_tensorarray::call(self, spacing, dim, edge_order);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor div_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::div_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & div__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::div__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor div_Tensor_mode_generated_plumbing(const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::div_Tensor_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, rounding_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & div__Tensor_mode_generated_plumbing(at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::div__Tensor_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim, rounding_mode);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor div_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::div_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & div__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::div__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor div_Scalar_mode_generated_plumbing(const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::div_Scalar_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, rounding_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & div__Scalar_mode_generated_plumbing(at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::div__Scalar_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other, rounding_mode);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor divide_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::divide_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & divide__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::divide__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor divide_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::divide_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & divide__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::divide__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor divide_Tensor_mode_generated_plumbing(const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::divide_Tensor_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, rounding_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & divide__Tensor_mode_generated_plumbing(at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::divide__Tensor_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim, rounding_mode);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor divide_Scalar_mode_generated_plumbing(const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::divide_Scalar_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, rounding_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & divide__Scalar_mode_generated_plumbing(at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::divide__Scalar_mode::call(self, other, rounding_mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other, rounding_mode);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor true_divide_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::true_divide_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & true_divide__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::true_divide__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor true_divide_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::true_divide_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & true_divide__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::true_divide__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor row_stack_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::row_stack::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_empty_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_empty::call(self, size, dtype, layout, device, pin_memory);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_empty_strided_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_empty_strided::call(self, size, stride, dtype, layout, device, pin_memory);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_full_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_full::call(self, size, fill_value, dtype, layout, device, pin_memory);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, fill_value, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_zeros_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_zeros::call(self, size, dtype, layout, device, pin_memory);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_ones_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_ones::call(self, size, dtype, layout, device, pin_memory);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
const at::Tensor & _resize_output__generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, at::Device device) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_resize_output_::call(self, size, device);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, size, device);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor empty_like_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::empty_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor erf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::erf::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & erf__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::erf_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor exp_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exp::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & exp__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exp_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor exp2_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exp2::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & exp2__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exp2_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor expm1_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::expm1::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & expm1__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::expm1_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor expand_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, bool implicit) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::expand::call(self, size, implicit);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, implicit);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor expand_as_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::expand_as::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flatten_using_ints_generated_plumbing(const at::Tensor & self, int64_t start_dim, int64_t end_dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::flatten_using_ints::call(self, start_dim, end_dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, start_dim, end_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flatten_named_out_dim_generated_plumbing(const at::Tensor & self, int64_t start_dim, int64_t end_dim, at::Dimname out_dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::flatten_named_out_dim::call(self, start_dim, end_dim, out_dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, start_dim, end_dim, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flatten_using_names_generated_plumbing(const at::Tensor & self, at::Dimname start_dim, at::Dimname end_dim, at::Dimname out_dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::flatten_using_names::call(self, start_dim, end_dim, out_dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, start_dim, end_dim, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flatten_DimnameList_generated_plumbing(const at::Tensor & self, at::DimnameList dims, at::Dimname out_dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::flatten_DimnameList::call(self, dims, out_dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unflatten_int_generated_plumbing(const at::Tensor & self, int64_t dim, c10::SymIntArrayRef sizes) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unflatten_int::call(self, dim, sizes);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, sizes);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unflatten_Dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, c10::SymIntArrayRef sizes, at::DimnameList names) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unflatten_Dimname::call(self, dim, sizes, names);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, sizes, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unfold_generated_plumbing(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unfold::call(self, dimension, size, step);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dimension, size, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unfold_backward_generated_plumbing(const at::Tensor & grad_in, c10::SymIntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_in, cur_level)) {
    return at::_ops::unfold_backward::call(grad_in, input_sizes, dim, size, step);
  }
  auto [grad_in_value, grad_in_bdim] = unwrapTensorAtLevel(grad_in, cur_level);
  auto results = batch_rule(grad_in_value, grad_in_bdim, input_sizes, dim, size, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fill_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fill_Scalar::call(self, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fill_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::fill_Tensor::call(self, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  auto results = batch_rule(self_value, self_bdim, value_value, value_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & fill__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fill__Scalar::call(self, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & fill__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::fill__Tensor::call(self, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  batch_rule(self_value, self_bdim, value_value, value_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor floor_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::floor::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & floor__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::floor_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor floor_divide_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::floor_divide::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & floor_divide__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::floor_divide__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor floor_divide_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::floor_divide_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & floor_divide__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::floor_divide__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor frac_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::frac::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & frac__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::frac_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor full_like_generated_plumbing(const at::Tensor & self, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::full_like::call(self, fill_value, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, fill_value, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gcd_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::gcd::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & gcd__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::gcd_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lcm_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::lcm::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & lcm__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::lcm_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _validate_compressed_sparse_indices_generated_plumbing(bool is_crow, const at::Tensor & compressed_idx, const at::Tensor & plain_idx, int64_t cdim, int64_t dim, int64_t nnz) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(compressed_idx, cur_level) && !isBatchedAtLevel(plain_idx, cur_level)) {
    return at::_ops::_validate_compressed_sparse_indices::call(is_crow, compressed_idx, plain_idx, cdim, dim, nnz);
  }
  auto [compressed_idx_value, compressed_idx_bdim] = unwrapTensorAtLevel(compressed_idx, cur_level);
  auto [plain_idx_value, plain_idx_bdim] = unwrapTensorAtLevel(plain_idx, cur_level);
  batch_rule(is_crow, compressed_idx_value, compressed_idx_bdim, plain_idx_value, plain_idx_bdim, cdim, dim, nnz);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_Tensor_generated_plumbing(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::index_Tensor::call(self, indices);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _unsafe_index_Tensor_generated_plumbing(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::_unsafe_index_Tensor::call(self, indices);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _unsafe_masked_index_generated_plumbing(const at::Tensor & self, const at::Tensor & mask, const c10::List<::std::optional<at::Tensor>> & indices, const at::Scalar & fill) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::_unsafe_masked_index::call(self, mask, indices, fill);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, indices, fill);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _unsafe_masked_index_put_accumulate_generated_plumbing(const at::Tensor & self, const at::Tensor & mask, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::_unsafe_masked_index_put_accumulate::call(self, mask, indices, values);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, indices, values_value, values_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_copy__generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_copy_::call(self, dim, index, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_copy_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_copy::call(self, dim, index, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_copy__dimname_generated_plumbing(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_copy__dimname::call(self, dim, index, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_copy_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_copy_dimname::call(self, dim, index, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_put__generated_plumbing(at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::index_put_::call(self, indices, values, accumulate);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  batch_rule(self_value, self_bdim, indices, values_value, values_bdim, accumulate);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_put_generated_plumbing(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::index_put::call(self, indices, values, accumulate);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices, values_value, values_bdim, accumulate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _unsafe_index_put_generated_plumbing(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::_unsafe_index_put::call(self, indices, values, accumulate);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices, values_value, values_bdim, accumulate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & _index_put_impl__generated_plumbing(at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::_index_put_impl_::call(self, indices, values, accumulate, unsafe);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  batch_rule(self_value, self_bdim, indices, values_value, values_bdim, accumulate, unsafe);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isclose_generated_plumbing(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::isclose::call(self, other, rtol, atol, equal_nan);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, rtol, atol, equal_nan);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isin_Tensor_Tensor_generated_plumbing(const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(elements, cur_level) && !isBatchedAtLevel(test_elements, cur_level)) {
    return at::_ops::isin_Tensor_Tensor::call(elements, test_elements, assume_unique, invert);
  }
  auto [elements_value, elements_bdim] = unwrapTensorAtLevel(elements, cur_level);
  auto [test_elements_value, test_elements_bdim] = unwrapTensorAtLevel(test_elements, cur_level);
  auto results = batch_rule(elements_value, elements_bdim, test_elements_value, test_elements_bdim, assume_unique, invert);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isin_Tensor_Scalar_generated_plumbing(const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(elements, cur_level)) {
    return at::_ops::isin_Tensor_Scalar::call(elements, test_element, assume_unique, invert);
  }
  auto [elements_value, elements_bdim] = unwrapTensorAtLevel(elements, cur_level);
  auto results = batch_rule(elements_value, elements_bdim, test_element, assume_unique, invert);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isin_Scalar_Tensor_generated_plumbing(const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(test_elements, cur_level)) {
    return at::_ops::isin_Scalar_Tensor::call(element, test_elements, assume_unique, invert);
  }
  auto [test_elements_value, test_elements_bdim] = unwrapTensorAtLevel(test_elements, cur_level);
  auto results = batch_rule(element, test_elements_value, test_elements_bdim, assume_unique, invert);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isnan_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isnan::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isreal_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isreal::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> kthvalue_generated_plumbing(const at::Tensor & self, c10::SymInt k, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::kthvalue::call(self, k, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> kthvalue_dimname_generated_plumbing(const at::Tensor & self, c10::SymInt k, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::kthvalue_dimname::call(self, k, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nan_to_num_generated_plumbing(const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nan_to_num::call(self, nan, posinf, neginf);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, nan, posinf, neginf);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & nan_to_num__generated_plumbing(at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nan_to_num_::call(self, nan, posinf, neginf);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, nan, posinf, neginf);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ldexp_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ldexp_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & ldexp__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ldexp_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor linspace_Tensor_Tensor_generated_plumbing(const at::Tensor & start, const at::Tensor & end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(start, cur_level) && !isBatchedAtLevel(end, cur_level)) {
    return at::_ops::linspace_Tensor_Tensor::call(start, end, steps, dtype, layout, device, pin_memory);
  }
  auto [start_value, start_bdim] = unwrapTensorAtLevel(start, cur_level);
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  auto results = batch_rule(start_value, start_bdim, end_value, end_bdim, steps, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor linspace_Tensor_Scalar_generated_plumbing(const at::Tensor & start, const at::Scalar & end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(start, cur_level)) {
    return at::_ops::linspace_Tensor_Scalar::call(start, end, steps, dtype, layout, device, pin_memory);
  }
  auto [start_value, start_bdim] = unwrapTensorAtLevel(start, cur_level);
  auto results = batch_rule(start_value, start_bdim, end, steps, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor linspace_Scalar_Tensor_generated_plumbing(const at::Scalar & start, const at::Tensor & end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(end, cur_level)) {
    return at::_ops::linspace_Scalar_Tensor::call(start, end, steps, dtype, layout, device, pin_memory);
  }
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  auto results = batch_rule(start, end_value, end_bdim, steps, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & log__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log10_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log10::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & log10__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log10_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log1p_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log1p::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & log1p__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log1p_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log2_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log2::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & log2__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log2_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logaddexp_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logaddexp::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logaddexp2_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logaddexp2::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logspace_Tensor_Tensor_generated_plumbing(const at::Tensor & start, const at::Tensor & end, int64_t steps, double base, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(start, cur_level) && !isBatchedAtLevel(end, cur_level)) {
    return at::_ops::logspace_Tensor_Tensor::call(start, end, steps, base, dtype, layout, device, pin_memory);
  }
  auto [start_value, start_bdim] = unwrapTensorAtLevel(start, cur_level);
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  auto results = batch_rule(start_value, start_bdim, end_value, end_bdim, steps, base, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logspace_Tensor_Scalar_generated_plumbing(const at::Tensor & start, const at::Scalar & end, int64_t steps, double base, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(start, cur_level)) {
    return at::_ops::logspace_Tensor_Scalar::call(start, end, steps, base, dtype, layout, device, pin_memory);
  }
  auto [start_value, start_bdim] = unwrapTensorAtLevel(start, cur_level);
  auto results = batch_rule(start_value, start_bdim, end, steps, base, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logspace_Scalar_Tensor_generated_plumbing(const at::Scalar & start, const at::Tensor & end, int64_t steps, double base, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(end, cur_level)) {
    return at::_ops::logspace_Scalar_Tensor::call(start, end, steps, base, dtype, layout, device, pin_memory);
  }
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  auto results = batch_rule(start, end_value, end_bdim, steps, base, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _logcumsumexp_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_logcumsumexp::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logcumsumexp_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logcumsumexp::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logcumsumexp_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logcumsumexp_dimname::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logsumexp_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logsumexp::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logsumexp_names_generated_plumbing(const at::Tensor & self, at::DimnameList dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logsumexp_names::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> _aminmax_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_aminmax::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> _aminmax_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_aminmax_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> aminmax_generated_plumbing(const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::aminmax::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> max_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::max_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> max_names_dim_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::max_names_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor value_selecting_reduction_backward_generated_plumbing(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::value_selecting_reduction_backward::call(grad, dim, indices, sizes, keepdim);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [indices_value, indices_bdim] = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, dim, indices_value, indices_bdim, sizes, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor amax_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::amax::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mean_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mean::call(self, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mean_dim_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mean_dim::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mean_names_dim_generated_plumbing(const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mean_names_dim::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nanmean_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nanmean::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor median_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::median::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> median_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::median_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> median_names_dim_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::median_names_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nanmedian_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nanmedian::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> nanmedian_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nanmedian_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> nanmedian_names_dim_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nanmedian_names_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> min_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::min_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> min_names_dim_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::min_names_dim::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor amin_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::amin::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> mode_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mode::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> mode_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mode_dimname::call(self, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::mul_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & mul__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::mul__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mul_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & mul__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mul__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor multiply_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::multiply_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & multiply__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::multiply__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor multiply_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::multiply_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & multiply__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::multiply__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor narrow_copy_generated_plumbing(const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::narrow_copy::call(self, dim, start, length);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, length);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor narrow_generated_plumbing(const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::narrow::call(self, dim, start, length);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, length);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor narrow_Tensor_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & start, c10::SymInt length) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(start, cur_level)) {
    return at::_ops::narrow_Tensor::call(self, dim, start, length);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [start_value, start_bdim] = unwrapTensorAtLevel(start, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start_value, start_bdim, length);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ones_like_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ones_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor permute_generated_plumbing(const at::Tensor & self, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::permute::call(self, dims);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor movedim_intlist_generated_plumbing(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::movedim_intlist::call(self, source, destination);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source, destination);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor movedim_int_generated_plumbing(const at::Tensor & self, int64_t source, int64_t destination) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::movedim_int::call(self, source, destination);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source, destination);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor moveaxis_intlist_generated_plumbing(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::moveaxis_intlist::call(self, source, destination);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source, destination);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor moveaxis_int_generated_plumbing(const at::Tensor & self, int64_t source, int64_t destination) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::moveaxis_int::call(self, source, destination);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source, destination);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor numpy_T_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::numpy_T::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor matrix_H_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::matrix_H::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mT_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mT::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mH_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mH::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor adjoint_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::adjoint::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pin_memory_generated_plumbing(const at::Tensor & self, ::std::optional<at::Device> device) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::pin_memory::call(self, device);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, device);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _pin_memory_generated_plumbing(const at::Tensor & self, ::std::optional<at::Device> device) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_pin_memory::call(self, device);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, device);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rad2deg_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rad2deg::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & rad2deg__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rad2deg_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor deg2rad_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::deg2rad::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & deg2rad__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::deg2rad_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rand_like_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rand_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rand_like_generator_generated_plumbing(const at::Tensor & self, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rand_like_generator::call(self, generator, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, generator, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randint_like_generated_plumbing(const at::Tensor & self, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::randint_like::call(self, high, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, high, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randint_like_generator_generated_plumbing(const at::Tensor & self, c10::SymInt high, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::randint_like_generator::call(self, high, generator, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, high, generator, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randint_like_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(high, cur_level)) {
    return at::_ops::randint_like_Tensor::call(self, high, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [high_value, high_bdim] = unwrapTensorAtLevel(high, cur_level);
  auto results = batch_rule(self_value, self_bdim, high_value, high_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randint_like_Tensor_generator_generated_plumbing(const at::Tensor & self, const at::Tensor & high, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(high, cur_level)) {
    return at::_ops::randint_like_Tensor_generator::call(self, high, generator, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [high_value, high_bdim] = unwrapTensorAtLevel(high, cur_level);
  auto results = batch_rule(self_value, self_bdim, high_value, high_bdim, generator, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randint_like_low_dtype_generated_plumbing(const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, low, high, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randint_like_low_generator_dtype_generated_plumbing(const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::randint_like_low_generator_dtype::call(self, low, high, generator, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, low, high, generator, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randn_like_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::randn_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randn_like_generator_generated_plumbing(const at::Tensor & self, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::randn_like_generator::call(self, generator, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, generator, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ravel_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ravel::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor reciprocal_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::reciprocal::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & reciprocal__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::reciprocal_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor neg_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::neg::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & neg__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::neg_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor negative_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::negative::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & negative__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::negative_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor repeat_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef repeats) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::repeat::call(self, repeats);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, repeats);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor repeat_interleave_Tensor_generated_plumbing(const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(repeats, cur_level)) {
    return at::_ops::repeat_interleave_Tensor::call(repeats, output_size);
  }
  auto [repeats_value, repeats_bdim] = unwrapTensorAtLevel(repeats, cur_level);
  auto results = batch_rule(repeats_value, repeats_bdim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor repeat_interleave_self_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(repeats, cur_level)) {
    return at::_ops::repeat_interleave_self_Tensor::call(self, repeats, dim, output_size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [repeats_value, repeats_bdim] = unwrapTensorAtLevel(repeats, cur_level);
  auto results = batch_rule(self_value, self_bdim, repeats_value, repeats_bdim, dim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor repeat_interleave_self_int_generated_plumbing(const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::repeat_interleave_self_int::call(self, repeats, dim, output_size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, repeats, dim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor reshape_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef shape) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::reshape::call(self, shape);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, shape);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _reshape_copy_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_reshape_copy::call(self, size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _reshape_alias_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_reshape_alias::call(self, size, stride);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor reshape_as_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::reshape_as::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor round_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::round::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & round__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::round_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor round_decimals_generated_plumbing(const at::Tensor & self, int64_t decimals) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::round_decimals::call(self, decimals);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, decimals);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & round__decimals_generated_plumbing(at::Tensor & self, int64_t decimals) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::round__decimals::call(self, decimals);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, decimals);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rsqrt_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rsqrt::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & rsqrt__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rsqrt_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor select_Dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, int64_t index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::select_Dimname::call(self, dim, index);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor select_int_generated_plumbing(const at::Tensor & self, int64_t dim, c10::SymInt index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::select_int::call(self, dim, index);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor select_backward_generated_plumbing(const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t dim, c10::SymInt index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::select_backward::call(grad_output, input_sizes, dim, index);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sigmoid_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sigmoid::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sigmoid__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sigmoid_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sin_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sin::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sin__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sin_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sinc_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sinc::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sinc__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sinc_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sinh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sinh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sinh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sinh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor detach_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::detach::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor slice_Tensor_generated_plumbing(const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::slice_Tensor::call(self, dim, start, end, step);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor slice_backward_generated_plumbing(const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t dim, c10::SymInt start, c10::SymInt end, c10::SymInt step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::slice_backward::call(grad_output, input_sizes, dim, start, end, step);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor slice_inverse_generated_plumbing(const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::slice_inverse::call(self, src, dim, start, end, step);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor slice_scatter_generated_plumbing(const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::slice_scatter::call(self, src, dim, start, end, step);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor select_scatter_generated_plumbing(const at::Tensor & self, const at::Tensor & src, int64_t dim, c10::SymInt index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::select_scatter::call(self, src, dim, index);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_scatter_generated_plumbing(const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::diagonal_scatter::call(self, src, offset, dim1, dim2);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor as_strided_scatter_generated_plumbing(const at::Tensor & self, const at::Tensor & src, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::as_strided_scatter::call(self, src, size, stride, storage_offset);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, size, stride, storage_offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> unsafe_split_Tensor_generated_plumbing(const at::Tensor & self, c10::SymInt split_size, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unsafe_split_Tensor::call(self, split_size, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_size, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> split_Tensor_generated_plumbing(const at::Tensor & self, c10::SymInt split_size, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::split_Tensor::call(self, split_size, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_size, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> split_sizes_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef split_size, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::split_sizes::call(self, split_size, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_size, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> unsafe_split_with_sizes_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unsafe_split_with_sizes::call(self, split_sizes, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_sizes, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> split_with_sizes_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::split_with_sizes::call(self, split_sizes, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_sizes, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> hsplit_int_generated_plumbing(const at::Tensor & self, int64_t sections) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hsplit_int::call(self, sections);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sections);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> hsplit_array_generated_plumbing(const at::Tensor & self, at::IntArrayRef indices) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hsplit_array::call(self, indices);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> vsplit_int_generated_plumbing(const at::Tensor & self, int64_t sections) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::vsplit_int::call(self, sections);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sections);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> vsplit_array_generated_plumbing(const at::Tensor & self, at::IntArrayRef indices) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::vsplit_array::call(self, indices);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> dsplit_int_generated_plumbing(const at::Tensor & self, int64_t sections) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::dsplit_int::call(self, sections);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sections);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> dsplit_array_generated_plumbing(const at::Tensor & self, at::IntArrayRef indices) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::dsplit_array::call(self, indices);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_dim_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze_dim::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze_dimname::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_dims_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze_dims::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _chunk_cat_generated_plumbing(at::TensorList tensors, int64_t dim, int64_t num_chunks) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::_chunk_cat::call(tensors, dim, num_chunks);
  }

  auto results = batch_rule(tensors, dim, num_chunks);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor stack_generated_plumbing(at::TensorList tensors, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::stack::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _stack_generated_plumbing(at::TensorList tensors, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::_stack::call(tensors, dim);
  }

  auto results = batch_rule(tensors, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hstack_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::hstack::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor vstack_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::vstack::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor dstack_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::dstack::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sum_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sum::call(self, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sum_dim_IntList_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sum_dim_IntList::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sum_dim_DimnameList_generated_plumbing(const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sum_dim_DimnameList::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nansum_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nansum::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hash_tensor_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, int64_t mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hash_tensor::call(self, dim, keepdim, mode);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sum_to_size_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sum_to_size::call(self, size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sqrt_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sqrt::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sqrt__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sqrt_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor square_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::square::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & square__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::square_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor std_generated_plumbing(const at::Tensor & self, bool unbiased) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std::call(self, unbiased);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, unbiased);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor std_dim_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor std_correction_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_correction::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> std_mean_generated_plumbing(const at::Tensor & self, bool unbiased) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_mean::call(self, unbiased);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, unbiased);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> std_mean_dim_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_mean_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> std_mean_correction_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_mean_correction::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> std_mean_names_dim_generated_plumbing(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_mean_names_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> std_mean_correction_names_generated_plumbing(const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_mean_correction_names::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor std_names_dim_generated_plumbing(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_names_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor std_correction_names_generated_plumbing(const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::std_correction_names::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor prod_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::prod::call(self, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor prod_dim_int_generated_plumbing(const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::prod_dim_int::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor prod_dim_Dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::prod_dim_Dimname::call(self, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor t_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::t::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tan_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tan::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & tan__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tan_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tanh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tanh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & tanh__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tanh_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tile_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tile::call(self, dims);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor transpose_int_generated_plumbing(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::transpose_int::call(self, dim0, dim1);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim0, dim1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor transpose_Dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::transpose_Dimname::call(self, dim0, dim1);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim0, dim1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flip_generated_plumbing(const at::Tensor & self, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::flip::call(self, dims);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fliplr_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fliplr::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flipud_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::flipud::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor roll_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef shifts, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::roll::call(self, shifts, dims);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, shifts, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rot90_generated_plumbing(const at::Tensor & self, int64_t k, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rot90::call(self, k, dims);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor trunc_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::trunc::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & trunc__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::trunc_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fix_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fix::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & fix__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fix_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor type_as_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::type_as::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> _unique_generated_plumbing(const at::Tensor & self, bool sorted, bool return_inverse) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_unique::call(self, sorted, return_inverse);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sorted, return_inverse);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_generated_plumbing(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unique_dim::call(self, dim, sorted, return_inverse, return_counts);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, sorted, return_inverse, return_counts);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive_generated_plumbing(const at::Tensor & self, bool return_inverse, bool return_counts, ::std::optional<int64_t> dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unique_consecutive::call(self, return_inverse, return_counts, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, return_inverse, return_counts, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_consecutive_generated_plumbing(const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unique_dim_consecutive::call(self, dim, return_inverse, return_counts);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, return_inverse, return_counts);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _unique2_generated_plumbing(const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_unique2::call(self, sorted, return_inverse, return_counts);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sorted, return_inverse, return_counts);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _unsafe_view_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_unsafe_view::call(self, size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unsqueeze_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unsqueeze::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor var_generated_plumbing(const at::Tensor & self, bool unbiased) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var::call(self, unbiased);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, unbiased);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor var_dim_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor var_correction_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_correction::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor var_names_dim_generated_plumbing(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_names_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor var_correction_names_generated_plumbing(const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_correction_names::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> var_mean_generated_plumbing(const at::Tensor & self, bool unbiased) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_mean::call(self, unbiased);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, unbiased);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> var_mean_dim_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_mean_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> var_mean_correction_generated_plumbing(const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_mean_correction::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> var_mean_names_dim_generated_plumbing(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_mean_names_dim::call(self, dim, unbiased, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> var_mean_correction_names_generated_plumbing(const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::var_mean_correction_names::call(self, dim, correction, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_as_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::view_as::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor where_self_generated_plumbing(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(condition, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::where_self::call(condition, self, other);
  }
  auto [condition_value, condition_bdim] = unwrapTensorAtLevel(condition, cur_level);
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor where_ScalarSelf_generated_plumbing(const at::Tensor & condition, const at::Scalar & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(condition, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::where_ScalarSelf::call(condition, self, other);
  }
  auto [condition_value, condition_bdim] = unwrapTensorAtLevel(condition, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor where_ScalarOther_generated_plumbing(const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(condition, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::where_ScalarOther::call(condition, self, other);
  }
  auto [condition_value, condition_bdim] = unwrapTensorAtLevel(condition, cur_level);
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor where_Scalar_generated_plumbing(const at::Tensor & condition, const at::Scalar & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(condition, cur_level)) {
    return at::_ops::where_Scalar::call(condition, self, other);
  }
  auto [condition_value, condition_bdim] = unwrapTensorAtLevel(condition, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> where_generated_plumbing(const at::Tensor & condition) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(condition, cur_level)) {
    return at::_ops::where::call(condition);
  }
  auto [condition_value, condition_bdim] = unwrapTensorAtLevel(condition, cur_level);
  auto results = batch_rule(condition_value, condition_bdim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor zeros_like_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::zeros_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor norm_ScalarOpt_dtype_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::norm_ScalarOpt_dtype::call(self, p, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor norm_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & p) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::norm_Scalar::call(self, p);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor norm_ScalarOpt_dim_dtype_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::norm_ScalarOpt_dim_dtype::call(self, p, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor norm_ScalarOpt_dim_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::norm_ScalarOpt_dim::call(self, p, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor norm_names_ScalarOpt_dim_dtype_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::norm_names_ScalarOpt_dim_dtype::call(self, p, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor norm_names_ScalarOpt_dim_generated_plumbing(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::norm_names_ScalarOpt_dim::call(self, p, dim, keepdim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> frexp_Tensor_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::frexp_Tensor::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clone_generated_plumbing(const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clone::call(self, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor positive_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::positive::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & zero__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::zero_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sub_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::sub_Tensor::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sub__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::sub__Tensor::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sub_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sub_Scalar::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sub__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sub__Scalar::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor subtract_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::subtract_Tensor::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & subtract__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::subtract__Tensor::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor subtract_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::subtract_Scalar::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & subtract__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::subtract__Scalar::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rsub_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::rsub_Tensor::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor heaviside_generated_plumbing(const at::Tensor & self, const at::Tensor & values) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::heaviside::call(self, values);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(self_value, self_bdim, values_value, values_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & heaviside__generated_plumbing(at::Tensor & self, const at::Tensor & values) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::heaviside_::call(self, values);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  batch_rule(self_value, self_bdim, values_value, values_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rsub_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rsub_Scalar::call(self, other, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> _to_cpu_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::_to_cpu::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_dense_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_dense::call(self, dtype, masked_grad);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, masked_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _to_dense_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_to_dense::call(self, dtype, masked_grad);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, masked_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_dense_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & input, ::std::optional<bool> masked_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(input, cur_level)) {
    return at::_ops::to_dense_backward::call(grad, input, masked_grad);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim, masked_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & _coalesced__generated_plumbing(at::Tensor & self, bool coalesced) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_coalesced_::call(self, coalesced);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, coalesced);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _indices_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_indices::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _values_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_values::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor indices_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::indices::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor values_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::values::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor crow_indices_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::crow_indices::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor col_indices_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::col_indices::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ccol_indices_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ccol_indices::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor row_indices_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::row_indices::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> unbind_int_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unbind_int::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> unbind_Dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unbind_Dimname::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_mkldnn_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_mkldnn::call(self, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _autocast_to_reduced_precision_generated_plumbing(const at::Tensor & self, bool cuda_enabled, bool cpu_enabled, at::ScalarType cuda_dtype, at::ScalarType cpu_dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_autocast_to_reduced_precision::call(self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _autocast_to_full_precision_generated_plumbing(const at::Tensor & self, bool cuda_enabled, bool cpu_enabled) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_autocast_to_full_precision::call(self, cuda_enabled, cpu_enabled);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, cuda_enabled, cpu_enabled);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _to_copy_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, bool non_blocking, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_to_copy::call(self, dtype, layout, device, pin_memory, non_blocking, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, non_blocking, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_dtype_layout_generated_plumbing(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_dtype_layout::call(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_device_generated_plumbing(const at::Tensor & self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_device::call(self, device, dtype, non_blocking, copy, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, device, dtype, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_dtype_generated_plumbing(const at::Tensor & self, at::ScalarType dtype, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_dtype::call(self, dtype, non_blocking, copy, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_other_generated_plumbing(const at::Tensor & self, const at::Tensor & other, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::to_other::call(self, other, non_blocking, copy, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> meshgrid_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::meshgrid::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> meshgrid_indexing_generated_plumbing(at::TensorList tensors, c10::string_view indexing) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::meshgrid_indexing::call(tensors, indexing);
  }

  auto results = batch_rule(tensors, indexing);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lift_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::lift::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lift_fresh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::lift_fresh::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lift_fresh_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::lift_fresh_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & masked_fill__Scalar_generated_plumbing(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_fill__Scalar::call(self, mask, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  batch_rule(self_value, self_bdim, mask_value, mask_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor masked_fill_Scalar_generated_plumbing(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_fill_Scalar::call(self, mask, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & masked_fill__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::masked_fill__Tensor::call(self, mask, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  batch_rule(self_value, self_bdim, mask_value, mask_bdim, value_value, value_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor masked_fill_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::masked_fill_Tensor::call(self, mask, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, value_value, value_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & masked_scatter__generated_plumbing(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::masked_scatter_::call(self, mask, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  batch_rule(self_value, self_bdim, mask_value, mask_bdim, source_value, source_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor masked_scatter_generated_plumbing(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::masked_scatter::call(self, mask, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, source_value, source_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor masked_scatter_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & mask, c10::SymIntArrayRef sizes) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_scatter_backward::call(grad_output, mask, sizes);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, mask_value, mask_bdim, sizes);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view::call(self, size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_dtype_generated_plumbing(const at::Tensor & self, at::ScalarType dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_dtype::call(self, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & put__generated_plumbing(at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::put_::call(self, index, source, accumulate);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  batch_rule(self_value, self_bdim, index_value, index_bdim, source_value, source_bdim, accumulate);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor put_generated_plumbing(const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::put::call(self, index, source, accumulate);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, index_value, index_bdim, source_value, source_bdim, accumulate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_add__generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_add_::call(self, dim, index, source, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_add_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_add::call(self, dim, index, source, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_add_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_add_dimname::call(self, dim, index, source, alpha);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_reduce__generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_reduce_::call(self, dim, index, source, reduce, include_self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim, reduce, include_self);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_reduce_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_reduce::call(self, dim, index, source, reduce, include_self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim, reduce, include_self);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_fill__int_Scalar_generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::index_fill__int_Scalar::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_fill_int_Scalar_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::index_fill_int_Scalar::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_fill__int_Tensor_generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::index_fill__int_Tensor::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value_value, value_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_fill_int_Tensor_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::index_fill_int_Tensor::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value_value, value_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_fill__Dimname_Scalar_generated_plumbing(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::index_fill__Dimname_Scalar::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & index_fill__Dimname_Tensor_generated_plumbing(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::index_fill__Dimname_Tensor::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value_value, value_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_fill_Dimname_Scalar_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::index_fill_Dimname_Scalar::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_fill_Dimname_Tensor_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(value, cur_level)) {
    return at::_ops::index_fill_Dimname_Tensor::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [value_value, value_bdim] = unwrapTensorAtLevel(value, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value_value, value_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_src_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_src::call(self, dim, index, src);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & scatter__src_generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter__src::call(self, dim, index, src);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_value_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::scatter_value::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & scatter__value_generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::scatter__value::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_reduce_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_reduce::call(self, dim, index, src, reduce);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim, reduce);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & scatter__reduce_generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter__reduce::call(self, dim, index, src, reduce);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim, reduce);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_value_reduce_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::scatter_value_reduce::call(self, dim, index, value, reduce);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value, reduce);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & scatter__value_reduce_generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::scatter__value_reduce::call(self, dim, index, value, reduce);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value, reduce);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_dimname_src_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_dimname_src::call(self, dim, index, src);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_dimname_value_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::scatter_dimname_value::call(self, dim, index, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_add_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_add::call(self, dim, index, src);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & scatter_add__generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_add_::call(self, dim, index, src);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_add_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_add_dimname::call(self, dim, index, src);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor scatter_reduce_two_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_reduce_two::call(self, dim, index, src, reduce, include_self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim, reduce, include_self);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & scatter_reduce__two_generated_plumbing(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_reduce__two::call(self, dim, index, src, reduce, include_self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto [src_value, src_bdim] = unwrapTensorAtLevel(src, cur_level);
  batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim, reduce, include_self);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & eq__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::eq__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & eq__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::eq__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_and_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_and_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_and_Scalar_Tensor_generated_plumbing(const at::Scalar & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_and_Scalar_Tensor::call(self, other);
  }
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_and_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_and_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_and__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_and__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_and__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_and__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __and___Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__and___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __and___Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__and___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __iand___Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__iand___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __iand___Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__iand___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_or_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_or_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_or_Scalar_Tensor_generated_plumbing(const at::Scalar & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_or_Scalar_Tensor::call(self, other);
  }
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_or_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_or_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_or__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_or__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_or__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_or__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __or___Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__or___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __or___Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__or___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __ior___Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__ior___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __ior___Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__ior___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_xor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_xor_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_xor_Scalar_Tensor_generated_plumbing(const at::Scalar & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_xor_Scalar_Tensor::call(self, other);
  }
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_xor_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_xor_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_xor__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_xor__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_xor__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_xor__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __xor___Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__xor___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __xor___Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__xor___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __ixor___Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__ixor___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __ixor___Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__ixor___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __lshift___Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__lshift___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __lshift___Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__lshift___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __ilshift___Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__ilshift___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __ilshift___Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__ilshift___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_left_shift_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_left_shift_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_left_shift__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_left_shift__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_left_shift_Tensor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_left_shift_Tensor_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_left_shift__Tensor_Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_left_shift__Tensor_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_left_shift_Scalar_Tensor_generated_plumbing(const at::Scalar & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_left_shift_Scalar_Tensor::call(self, other);
  }
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __rshift___Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__rshift___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __rshift___Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__rshift___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __irshift___Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__irshift___Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & __irshift___Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__irshift___Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_right_shift_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_right_shift_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_right_shift__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_right_shift__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_right_shift_Tensor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_right_shift_Tensor_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & bitwise_right_shift__Tensor_Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_right_shift__Tensor_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_right_shift_Scalar_Tensor_generated_plumbing(const at::Scalar & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_right_shift_Scalar_Tensor::call(self, other);
  }
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & tril__generated_plumbing(at::Tensor & self, c10::SymInt diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tril_::call(self, diagonal);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, diagonal);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & triu__generated_plumbing(at::Tensor & self, c10::SymInt diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::triu_::call(self, diagonal);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, diagonal);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & lerp__Scalar_generated_plumbing(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(end, cur_level)) {
    return at::_ops::lerp__Scalar::call(self, end, weight);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  batch_rule(self_value, self_bdim, end_value, end_bdim, weight);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & lerp__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(end, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::lerp__Tensor::call(self, end, weight);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  auto [weight_value, weight_bdim] = unwrapTensorAtLevel(weight, cur_level);
  batch_rule(self_value, self_bdim, end_value, end_bdim, weight_value, weight_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & random__from_generated_plumbing(at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::random__from::call(self, from, to, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, from, to, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & random__to_generated_plumbing(at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::random__to::call(self, to, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, to, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & random__generated_plumbing(at::Tensor & self, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::random_::call(self, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & uniform__generated_plumbing(at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::uniform_::call(self, from, to, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, from, to, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & exponential__generated_plumbing(at::Tensor & self, double lambd, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exponential_::call(self, lambd, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, lambd, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diag_generated_plumbing(const at::Tensor & self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diag::call(self, diagonal);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor triu_generated_plumbing(const at::Tensor & self, c10::SymInt diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::triu::call(self, diagonal);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tril_generated_plumbing(const at::Tensor & self, c10::SymInt diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tril::call(self, diagonal);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor trace_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::trace::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor trace_backward_generated_plumbing(const at::Tensor & grad, c10::SymIntArrayRef sizes) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level)) {
    return at::_ops::trace_backward::call(grad, sizes);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, sizes);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ne_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ne_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ne_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ne_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & ne__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ne__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & ne__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ne__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor not_equal_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::not_equal_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor not_equal_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::not_equal_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & not_equal__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::not_equal__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & not_equal__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::not_equal__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor eq_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::eq_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor eq_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::eq_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ge_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ge_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ge_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ge_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & ge__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ge__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & ge__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ge__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor greater_equal_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::greater_equal_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor greater_equal_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::greater_equal_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & greater_equal__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::greater_equal__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & greater_equal__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::greater_equal__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor le_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::le_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor le_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::le_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & le__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::le__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & le__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::le__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor less_equal_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::less_equal_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor less_equal_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::less_equal_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & less_equal__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::less_equal__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & less_equal__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::less_equal__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gt_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gt_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gt_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::gt_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & gt__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gt__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & gt__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::gt__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor greater_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::greater_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor greater_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::greater_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & greater__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::greater__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & greater__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::greater__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lt_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::lt_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lt_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::lt_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & lt__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::lt__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & lt__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::lt__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor less_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::less_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor less_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::less_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & less__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::less__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & less__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::less__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor take_generated_plumbing(const at::Tensor & self, const at::Tensor & index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::take::call(self, index);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor take_along_dim_generated_plumbing(const at::Tensor & self, const at::Tensor & indices, ::std::optional<int64_t> dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::take_along_dim::call(self, indices, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [indices_value, indices_bdim] = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices_value, indices_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_select_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::index_select::call(self, dim, index);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_select_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::index_select_dimname::call(self, dim, index);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_select_backward_generated_plumbing(const at::Tensor & grad, c10::SymIntArrayRef self_sizes, int64_t dim, const at::Tensor & index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::index_select_backward::call(grad, self_sizes, dim, index);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_sizes, dim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor masked_select_generated_plumbing(const at::Tensor & self, const at::Tensor & mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_select::call(self, mask);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor masked_select_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_select_backward::call(grad, input, mask);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [mask_value, mask_bdim] = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nonzero_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nonzero::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nonzero_static_generated_plumbing(const at::Tensor & self, c10::SymInt size, int64_t fill_value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nonzero_static::call(self, size, fill_value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, fill_value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> nonzero_numpy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nonzero_numpy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor argwhere_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::argwhere::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gather_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::gather::call(self, dim, index, sparse_grad);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, sparse_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gather_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::gather_backward::call(grad, self, dim, index, sparse_grad);
  }
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_value, self_bdim, dim, index_value, index_bdim, sparse_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gather_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::gather_dimname::call(self, dim, index, sparse_grad);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [index_value, index_bdim] = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, sparse_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor addcmul_generated_plumbing(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(tensor1, cur_level) && !isBatchedAtLevel(tensor2, cur_level)) {
    return at::_ops::addcmul::call(self, tensor1, tensor2, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [tensor1_value, tensor1_bdim] = unwrapTensorAtLevel(tensor1, cur_level);
  auto [tensor2_value, tensor2_bdim] = unwrapTensorAtLevel(tensor2, cur_level);
  auto results = batch_rule(self_value, self_bdim, tensor1_value, tensor1_bdim, tensor2_value, tensor2_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & addcmul__generated_plumbing(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(tensor1, cur_level) && !isBatchedAtLevel(tensor2, cur_level)) {
    return at::_ops::addcmul_::call(self, tensor1, tensor2, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [tensor1_value, tensor1_bdim] = unwrapTensorAtLevel(tensor1, cur_level);
  auto [tensor2_value, tensor2_bdim] = unwrapTensorAtLevel(tensor2, cur_level);
  batch_rule(self_value, self_bdim, tensor1_value, tensor1_bdim, tensor2_value, tensor2_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor addcdiv_generated_plumbing(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(tensor1, cur_level) && !isBatchedAtLevel(tensor2, cur_level)) {
    return at::_ops::addcdiv::call(self, tensor1, tensor2, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [tensor1_value, tensor1_bdim] = unwrapTensorAtLevel(tensor1, cur_level);
  auto [tensor2_value, tensor2_bdim] = unwrapTensorAtLevel(tensor2, cur_level);
  auto results = batch_rule(self_value, self_bdim, tensor1_value, tensor1_bdim, tensor2_value, tensor2_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & addcdiv__generated_plumbing(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(tensor1, cur_level) && !isBatchedAtLevel(tensor2, cur_level)) {
    return at::_ops::addcdiv_::call(self, tensor1, tensor2, value);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [tensor1_value, tensor1_bdim] = unwrapTensorAtLevel(tensor1, cur_level);
  auto [tensor2_value, tensor2_bdim] = unwrapTensorAtLevel(tensor2, cur_level);
  batch_rule(self_value, self_bdim, tensor1_value, tensor1_bdim, tensor2_value, tensor2_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor swapaxes_generated_plumbing(const at::Tensor & self, int64_t axis0, int64_t axis1) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::swapaxes::call(self, axis0, axis1);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, axis0, axis1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor swapdims_generated_plumbing(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::swapdims::call(self, dim0, dim1);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim0, dim1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sign_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sign::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sign__generated_plumbing(at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sign_::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor signbit_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::signbit::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & atan2__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::atan2_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atan2_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::atan2::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor arctan2_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::arctan2::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & arctan2__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::arctan2_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lerp_Scalar_generated_plumbing(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(end, cur_level)) {
    return at::_ops::lerp_Scalar::call(self, end, weight);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  auto results = batch_rule(self_value, self_bdim, end_value, end_bdim, weight);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lerp_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(end, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::lerp_Tensor::call(self, end, weight);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [end_value, end_bdim] = unwrapTensorAtLevel(end, cur_level);
  auto [weight_value, weight_bdim] = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(self_value, self_bdim, end_value, end_bdim, weight_value, weight_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fmod_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fmod_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & fmod__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fmod__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fmod_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::fmod_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & fmod__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::fmod__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hypot_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::hypot::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & hypot__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::hypot_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nextafter_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::nextafter::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & nextafter__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::nextafter_::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor remainder_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::remainder_Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & remainder__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::remainder__Scalar::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor remainder_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::remainder_Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & remainder__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::remainder__Tensor::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor remainder_Scalar_Tensor_generated_plumbing(const at::Scalar & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(other, cur_level)) {
    return at::_ops::remainder_Scalar_Tensor::call(self, other);
  }
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor min_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::min::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fmin_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::fmin::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor max_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::max::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fmax_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::fmax::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor maximum_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::maximum::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor max_other_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::max_other::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor minimum_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::minimum::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor min_other_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::min_other::call(self, other);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor quantile_generated_plumbing(const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(q, cur_level)) {
    return at::_ops::quantile::call(self, q, dim, keepdim, interpolation);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [q_value, q_bdim] = unwrapTensorAtLevel(q, cur_level);
  auto results = batch_rule(self_value, self_bdim, q_value, q_bdim, dim, keepdim, interpolation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor quantile_scalar_generated_plumbing(const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::quantile_scalar::call(self, q, dim, keepdim, interpolation);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, q, dim, keepdim, interpolation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nanquantile_generated_plumbing(const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(q, cur_level)) {
    return at::_ops::nanquantile::call(self, q, dim, keepdim, interpolation);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [q_value, q_bdim] = unwrapTensorAtLevel(q, cur_level);
  auto results = batch_rule(self_value, self_bdim, q_value, q_bdim, dim, keepdim, interpolation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor nanquantile_scalar_generated_plumbing(const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nanquantile_scalar::call(self, q, dim, keepdim, interpolation);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, q, dim, keepdim, interpolation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> sort_generated_plumbing(const at::Tensor & self, int64_t dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sort::call(self, dim, descending);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> sort_stable_generated_plumbing(const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sort_stable::call(self, stable, dim, descending);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, stable, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> sort_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sort_dimname::call(self, dim, descending);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> sort_dimname_stable_generated_plumbing(const at::Tensor & self, ::std::optional<bool> stable, at::Dimname dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sort_dimname_stable::call(self, stable, dim, descending);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, stable, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor msort_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::msort::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor argsort_generated_plumbing(const at::Tensor & self, int64_t dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::argsort::call(self, dim, descending);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor argsort_stable_generated_plumbing(const at::Tensor & self, bool stable, int64_t dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::argsort_stable::call(self, stable, dim, descending);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, stable, dim, descending);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor argsort_dimname_generated_plumbing(const at::Tensor & self, at::Dimname dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::argsort_dimname::call(self, dim, descending);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> topk_generated_plumbing(const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::topk::call(self, k, dim, largest, sorted);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dim, largest, sorted);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor all_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::all::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor any_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::any::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pow_Tensor_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(exponent, cur_level)) {
    return at::_ops::pow_Tensor_Tensor::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [exponent_value, exponent_bdim] = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pow_Scalar_generated_plumbing(const at::Scalar & self, const at::Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(exponent, cur_level)) {
    return at::_ops::pow_Scalar::call(self, exponent);
  }
  auto [exponent_value, exponent_bdim] = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pow_Tensor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::pow_Tensor_Scalar::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & pow__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::pow__Scalar::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, exponent);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & pow__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(exponent, cur_level)) {
    return at::_ops::pow__Tensor::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [exponent_value, exponent_bdim] = unwrapTensorAtLevel(exponent, cur_level);
  batch_rule(self_value, self_bdim, exponent_value, exponent_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor float_power_Tensor_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(exponent, cur_level)) {
    return at::_ops::float_power_Tensor_Tensor::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [exponent_value, exponent_bdim] = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor float_power_Scalar_generated_plumbing(const at::Scalar & self, const at::Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(exponent, cur_level)) {
    return at::_ops::float_power_Scalar::call(self, exponent);
  }
  auto [exponent_value, exponent_bdim] = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor float_power_Tensor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::float_power_Tensor_Scalar::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & float_power__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::float_power__Scalar::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, exponent);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & float_power__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(exponent, cur_level)) {
    return at::_ops::float_power__Tensor::call(self, exponent);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [exponent_value, exponent_bdim] = unwrapTensorAtLevel(exponent, cur_level);
  batch_rule(self_value, self_bdim, exponent_value, exponent_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & normal__generated_plumbing(at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::normal_::call(self, mean, std, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, mean, std, generator);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor normal_functional_generated_plumbing(const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::normal_functional::call(self, mean, std, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, mean, std, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor normal_Tensor_float_generated_plumbing(const at::Tensor & mean, double std, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(mean, cur_level)) {
    return at::_ops::normal_Tensor_float::call(mean, std, generator);
  }
  auto [mean_value, mean_bdim] = unwrapTensorAtLevel(mean, cur_level);
  auto results = batch_rule(mean_value, mean_bdim, std, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor normal_float_Tensor_generated_plumbing(double mean, const at::Tensor & std, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(std, cur_level)) {
    return at::_ops::normal_float_Tensor::call(mean, std, generator);
  }
  auto [std_value, std_bdim] = unwrapTensorAtLevel(std, cur_level);
  auto results = batch_rule(mean, std_value, std_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor normal_Tensor_Tensor_generated_plumbing(const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(mean, cur_level) && !isBatchedAtLevel(std, cur_level)) {
    return at::_ops::normal_Tensor_Tensor::call(mean, std, generator);
  }
  auto [mean_value, mean_bdim] = unwrapTensorAtLevel(mean, cur_level);
  auto [std_value, std_bdim] = unwrapTensorAtLevel(std, cur_level);
  auto results = batch_rule(mean_value, mean_bdim, std_value, std_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor alias_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::alias::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sigmoid_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & output) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::sigmoid_backward::call(grad_output, output);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto [output_value, output_bdim] = unwrapTensorAtLevel(output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tanh_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & output) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::tanh_backward::call(grad_output, output);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto [output_value, output_bdim] = unwrapTensorAtLevel(output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor column_stack_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::column_stack::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isfinite_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isfinite::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isinf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isinf::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void record_stream_generated_plumbing(at::Tensor & self, at::Stream s) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::record_stream::call(self, s);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, s);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isposinf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isposinf::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isneginf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isneginf::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _remove_batch_dim_generated_plumbing(const at::Tensor & self, int64_t level, c10::SymInt batch_size, int64_t out_dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_remove_batch_dim::call(self, level, batch_size, out_dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, level, batch_size, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flatten_dense_tensors_generated_plumbing(at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::flatten_dense_tensors::call(tensors);
  }

  auto results = batch_rule(tensors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> unflatten_dense_tensors_generated_plumbing(const at::Tensor & flat, at::TensorList tensors) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(flat, cur_level) && !isBatchedAtLevel(tensors, cur_level)) {
    return at::_ops::unflatten_dense_tensors::call(flat, tensors);
  }
  auto [flat_value, flat_bdim] = unwrapTensorAtLevel(flat, cur_level);
  auto results = batch_rule(flat_value, flat_bdim, tensors);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _fw_primal_copy_generated_plumbing(const at::Tensor & self, int64_t level) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_fw_primal_copy::call(self, level);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, level);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _make_dual_copy_generated_plumbing(const at::Tensor & primal, const at::Tensor & tangent, int64_t level) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(primal, cur_level) && !isBatchedAtLevel(tangent, cur_level)) {
    return at::_ops::_make_dual_copy::call(primal, tangent, level);
  }
  auto [primal_value, primal_bdim] = unwrapTensorAtLevel(primal, cur_level);
  auto [tangent_value, tangent_bdim] = unwrapTensorAtLevel(tangent, cur_level);
  auto results = batch_rule(primal_value, primal_bdim, tangent_value, tangent_bdim, level);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_as_real_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_as_real_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_as_complex_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_as_complex_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _conj_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_conj_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _neg_view_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_neg_view_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor as_strided_copy_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::as_strided_copy::call(self, size, stride, storage_offset);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride, storage_offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_copy_generated_plumbing(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diagonal_copy::call(self, offset, dim1, dim2);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor expand_copy_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, bool implicit) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::expand_copy::call(self, size, implicit);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, implicit);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor permute_copy_generated_plumbing(const at::Tensor & self, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::permute_copy::call(self, dims);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _reshape_alias_copy_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_reshape_alias_copy::call(self, size, stride);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor select_copy_int_generated_plumbing(const at::Tensor & self, int64_t dim, c10::SymInt index) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::select_copy_int::call(self, dim, index);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor detach_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::detach_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor slice_copy_Tensor_generated_plumbing(const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::slice_copy_Tensor::call(self, dim, start, end, step);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> split_copy_Tensor_generated_plumbing(const at::Tensor & self, c10::SymInt split_size, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::split_copy_Tensor::call(self, split_size, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_size, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> split_with_sizes_copy_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::split_with_sizes_copy::call(self, split_sizes, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_sizes, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_copy_dim_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze_copy_dim::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_copy_dims_generated_plumbing(const at::Tensor & self, at::IntArrayRef dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze_copy_dims::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor t_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::t_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor transpose_copy_int_generated_plumbing(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::transpose_copy_int::call(self, dim0, dim1);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim0, dim1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unsqueeze_copy_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unsqueeze_copy::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _indices_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_indices_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _values_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_values_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor indices_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::indices_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor values_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::values_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor crow_indices_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::crow_indices_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor col_indices_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::col_indices_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ccol_indices_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ccol_indices_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor row_indices_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::row_indices_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> unbind_copy_int_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unbind_copy_int::call(self, dim);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void unbind_copy_int_out_generated_plumbing(const at::Tensor & self, int64_t dim, at::TensorList out) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(out, cur_level)) {
    return at::_ops::unbind_copy_int_out::call(self, dim, out);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, dim, out);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void split_copy_Tensor_out_generated_plumbing(const at::Tensor & self, c10::SymInt split_size, int64_t dim, at::TensorList out) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(out, cur_level)) {
    return at::_ops::split_copy_Tensor_out::call(self, split_size, dim, out);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, split_size, dim, out);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void split_with_sizes_copy_out_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim, at::TensorList out) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(out, cur_level)) {
    return at::_ops::split_with_sizes_copy_out::call(self, split_sizes, dim, out);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, split_sizes, dim, out);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_copy_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_copy::call(self, size);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_copy_dtype_generated_plumbing(const at::Tensor & self, at::ScalarType dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_copy_dtype::call(self, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unfold_copy_generated_plumbing(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unfold_copy::call(self, dimension, size, step);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dimension, size, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor alias_copy_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::alias_copy::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _propagate_xla_data_generated_plumbing(const at::Tensor & input, const at::Tensor & output) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::_propagate_xla_data::call(input, output);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [output_value, output_bdim] = unwrapTensorAtLevel(output, cur_level);
  batch_rule(input_value, input_bdim, output_value, output_bdim);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor neg_mod_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::neg_mod::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & neg_mod__generated_plumbing(at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::neg_mod_::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor add_mod_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::add_mod::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & add_mod__generated_plumbing(at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::add_mod_::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sub_mod_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::sub_mod::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sub_mod__generated_plumbing(at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::sub_mod_::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_mod_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, const at::Tensor & barret_mu, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level) && !isBatchedAtLevel(barret_mu, cur_level)) {
    return at::_ops::mul_mod::call(self, in_b, mod, barret_mu, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto [barret_mu_value, barret_mu_bdim] = unwrapTensorAtLevel(barret_mu, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, barret_mu_value, barret_mu_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & mul_mod__generated_plumbing(at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, const at::Tensor & barret_mu, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level) && !isBatchedAtLevel(barret_mu, cur_level)) {
    return at::_ops::mul_mod_::call(self, in_b, mod, barret_mu, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto [barret_mu_value, barret_mu_bdim] = unwrapTensorAtLevel(barret_mu, cur_level);
  batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, barret_mu_value, barret_mu_bdim, cur_limbs);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor add_scalar_mod_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::add_scalar_mod::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & add_scalar_mod__generated_plumbing(at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::add_scalar_mod_::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sub_scalar_mod_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::sub_scalar_mod::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sub_scalar_mod__generated_plumbing(at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::sub_scalar_mod_::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_scalar_mod_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, const at::Tensor & barret_mu, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level) && !isBatchedAtLevel(barret_mu, cur_level)) {
    return at::_ops::mul_scalar_mod::call(self, in_b, mod, barret_mu, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto [barret_mu_value, barret_mu_bdim] = unwrapTensorAtLevel(barret_mu, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, barret_mu_value, barret_mu_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & mul_scalar_mod__generated_plumbing(at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, const at::Tensor & barret_mu, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level) && !isBatchedAtLevel(barret_mu, cur_level)) {
    return at::_ops::mul_scalar_mod_::call(self, in_b, mod, barret_mu, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto [barret_mu_value, barret_mu_bdim] = unwrapTensorAtLevel(barret_mu, cur_level);
  batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, barret_mu_value, barret_mu_bdim, cur_limbs);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_pt_broadcast_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, const at::Tensor & barret_mu, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level) && !isBatchedAtLevel(barret_mu, cur_level)) {
    return at::_ops::mul_pt_broadcast::call(self, in_b, mod, barret_mu, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto [barret_mu_value, barret_mu_bdim] = unwrapTensorAtLevel(barret_mu, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, barret_mu_value, barret_mu_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_pt_pairwise_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, const at::Tensor & barret_mu, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level) && !isBatchedAtLevel(barret_mu, cur_level)) {
    return at::_ops::mul_pt_pairwise::call(self, in_b, mod, barret_mu, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto [barret_mu_value, barret_mu_bdim] = unwrapTensorAtLevel(barret_mu, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, barret_mu_value, barret_mu_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor add_pt_broadcast_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::add_pt_broadcast::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor add_pt_pairwise_generated_plumbing(const at::Tensor & self, const at::Tensor & in_b, const at::Tensor & mod, int64_t cur_limbs) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(in_b, cur_level) && !isBatchedAtLevel(mod, cur_level)) {
    return at::_ops::add_pt_pairwise::call(self, in_b, mod, cur_limbs);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [in_b_value, in_b_bdim] = unwrapTensorAtLevel(in_b, cur_level);
  auto [mod_value, mod_bdim] = unwrapTensorAtLevel(mod, cur_level);
  auto results = batch_rule(self_value, self_bdim, in_b_value, in_b_bdim, mod_value, mod_bdim, cur_limbs);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor modup_generated_plumbing(const at::Tensor & input, int64_t curr_limbs, int64_t L, int64_t beta, int64_t degree, int64_t alpha, const at::Tensor & hat_inverse_vec, const at::Tensor & hat_inverse_vec_shoup, const at::Tensor & prod_q_i_mod_q_j, const at::Tensor & primes, const at::Tensor & barret_ratio, const at::Tensor & barret_k, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots, const at::Tensor & inverse_power_of_roots_div_two, const at::Tensor & inverse_scaled_power_of_roots_div_two) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(hat_inverse_vec, cur_level) && !isBatchedAtLevel(hat_inverse_vec_shoup, cur_level) && !isBatchedAtLevel(prod_q_i_mod_q_j, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(barret_ratio, cur_level) && !isBatchedAtLevel(barret_k, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level) && !isBatchedAtLevel(inverse_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(inverse_scaled_power_of_roots_div_two, cur_level)) {
    return at::_ops::modup::call(input, curr_limbs, L, beta, degree, alpha, hat_inverse_vec, hat_inverse_vec_shoup, prod_q_i_mod_q_j, primes, barret_ratio, barret_k, power_of_roots_shoup, power_of_roots, inverse_power_of_roots_div_two, inverse_scaled_power_of_roots_div_two);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [hat_inverse_vec_value, hat_inverse_vec_bdim] = unwrapTensorAtLevel(hat_inverse_vec, cur_level);
  auto [hat_inverse_vec_shoup_value, hat_inverse_vec_shoup_bdim] = unwrapTensorAtLevel(hat_inverse_vec_shoup, cur_level);
  auto [prod_q_i_mod_q_j_value, prod_q_i_mod_q_j_bdim] = unwrapTensorAtLevel(prod_q_i_mod_q_j, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [barret_ratio_value, barret_ratio_bdim] = unwrapTensorAtLevel(barret_ratio, cur_level);
  auto [barret_k_value, barret_k_bdim] = unwrapTensorAtLevel(barret_k, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto [inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_power_of_roots_div_two, cur_level);
  auto [inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_scaled_power_of_roots_div_two, cur_level);
  auto results = batch_rule(input_value, input_bdim, curr_limbs, L, beta, degree, alpha, hat_inverse_vec_value, hat_inverse_vec_bdim, hat_inverse_vec_shoup_value, hat_inverse_vec_shoup_bdim, prod_q_i_mod_q_j_value, prod_q_i_mod_q_j_bdim, primes_value, primes_bdim, barret_ratio_value, barret_ratio_bdim, barret_k_value, barret_k_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim, inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim, inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor moddown_generated_plumbing(const at::Tensor & input, int64_t curr_limbs, int64_t L, int64_t sizeP, int64_t N, int64_t logN, const at::Tensor & hat_inverse_vec_moddown, const at::Tensor & hat_inverse_vec_shoup_moddown, const at::Tensor & prod_q_i_mod_q_j_moddown, const at::Tensor & prod_inv_moddown, const at::Tensor & prod_inv_shoup_moddown, const at::Tensor & primes, const at::Tensor & barret_ratio, const at::Tensor & barret_k, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots, const at::Tensor & inverse_power_of_roots_div_two, const at::Tensor & inverse_scaled_power_of_roots_div_two) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(hat_inverse_vec_moddown, cur_level) && !isBatchedAtLevel(hat_inverse_vec_shoup_moddown, cur_level) && !isBatchedAtLevel(prod_q_i_mod_q_j_moddown, cur_level) && !isBatchedAtLevel(prod_inv_moddown, cur_level) && !isBatchedAtLevel(prod_inv_shoup_moddown, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(barret_ratio, cur_level) && !isBatchedAtLevel(barret_k, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level) && !isBatchedAtLevel(inverse_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(inverse_scaled_power_of_roots_div_two, cur_level)) {
    return at::_ops::moddown::call(input, curr_limbs, L, sizeP, N, logN, hat_inverse_vec_moddown, hat_inverse_vec_shoup_moddown, prod_q_i_mod_q_j_moddown, prod_inv_moddown, prod_inv_shoup_moddown, primes, barret_ratio, barret_k, power_of_roots_shoup, power_of_roots, inverse_power_of_roots_div_two, inverse_scaled_power_of_roots_div_two);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [hat_inverse_vec_moddown_value, hat_inverse_vec_moddown_bdim] = unwrapTensorAtLevel(hat_inverse_vec_moddown, cur_level);
  auto [hat_inverse_vec_shoup_moddown_value, hat_inverse_vec_shoup_moddown_bdim] = unwrapTensorAtLevel(hat_inverse_vec_shoup_moddown, cur_level);
  auto [prod_q_i_mod_q_j_moddown_value, prod_q_i_mod_q_j_moddown_bdim] = unwrapTensorAtLevel(prod_q_i_mod_q_j_moddown, cur_level);
  auto [prod_inv_moddown_value, prod_inv_moddown_bdim] = unwrapTensorAtLevel(prod_inv_moddown, cur_level);
  auto [prod_inv_shoup_moddown_value, prod_inv_shoup_moddown_bdim] = unwrapTensorAtLevel(prod_inv_shoup_moddown, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [barret_ratio_value, barret_ratio_bdim] = unwrapTensorAtLevel(barret_ratio, cur_level);
  auto [barret_k_value, barret_k_bdim] = unwrapTensorAtLevel(barret_k, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto [inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_power_of_roots_div_two, cur_level);
  auto [inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_scaled_power_of_roots_div_two, cur_level);
  auto results = batch_rule(input_value, input_bdim, curr_limbs, L, sizeP, N, logN, hat_inverse_vec_moddown_value, hat_inverse_vec_moddown_bdim, hat_inverse_vec_shoup_moddown_value, hat_inverse_vec_shoup_moddown_bdim, prod_q_i_mod_q_j_moddown_value, prod_q_i_mod_q_j_moddown_bdim, prod_inv_moddown_value, prod_inv_moddown_bdim, prod_inv_shoup_moddown_value, prod_inv_shoup_moddown_bdim, primes_value, primes_bdim, barret_ratio_value, barret_ratio_bdim, barret_k_value, barret_k_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim, inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim, inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor drop_last_element_and_scale_generated_plumbing(const at::Tensor & input, int64_t curr_limbs, int64_t l, int64_t L, int64_t N, int64_t old_prime, const at::Tensor & primes, const at::Tensor & switch_modulus_map, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots, const at::Tensor & inverse_power_of_roots_div_two, const at::Tensor & inverse_scaled_power_of_roots_div_two, const at::Tensor & qlql_inv_mod_ql_div_ql_mod_q, const at::Tensor & qlql_inv_mod_ql_div_ql_mod_q_shoup, const at::Tensor & q_inv_mod_q, const at::Tensor & q_inv_mod_q_shoup) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(switch_modulus_map, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level) && !isBatchedAtLevel(inverse_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(inverse_scaled_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(qlql_inv_mod_ql_div_ql_mod_q, cur_level) && !isBatchedAtLevel(qlql_inv_mod_ql_div_ql_mod_q_shoup, cur_level) && !isBatchedAtLevel(q_inv_mod_q, cur_level) && !isBatchedAtLevel(q_inv_mod_q_shoup, cur_level)) {
    return at::_ops::drop_last_element_and_scale::call(input, curr_limbs, l, L, N, old_prime, primes, switch_modulus_map, power_of_roots_shoup, power_of_roots, inverse_power_of_roots_div_two, inverse_scaled_power_of_roots_div_two, qlql_inv_mod_ql_div_ql_mod_q, qlql_inv_mod_ql_div_ql_mod_q_shoup, q_inv_mod_q, q_inv_mod_q_shoup);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [switch_modulus_map_value, switch_modulus_map_bdim] = unwrapTensorAtLevel(switch_modulus_map, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto [inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_power_of_roots_div_two, cur_level);
  auto [inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_scaled_power_of_roots_div_two, cur_level);
  auto [qlql_inv_mod_ql_div_ql_mod_q_value, qlql_inv_mod_ql_div_ql_mod_q_bdim] = unwrapTensorAtLevel(qlql_inv_mod_ql_div_ql_mod_q, cur_level);
  auto [qlql_inv_mod_ql_div_ql_mod_q_shoup_value, qlql_inv_mod_ql_div_ql_mod_q_shoup_bdim] = unwrapTensorAtLevel(qlql_inv_mod_ql_div_ql_mod_q_shoup, cur_level);
  auto [q_inv_mod_q_value, q_inv_mod_q_bdim] = unwrapTensorAtLevel(q_inv_mod_q, cur_level);
  auto [q_inv_mod_q_shoup_value, q_inv_mod_q_shoup_bdim] = unwrapTensorAtLevel(q_inv_mod_q_shoup, cur_level);
  auto results = batch_rule(input_value, input_bdim, curr_limbs, l, L, N, old_prime, primes_value, primes_bdim, switch_modulus_map_value, switch_modulus_map_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim, inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim, inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim, qlql_inv_mod_ql_div_ql_mod_q_value, qlql_inv_mod_ql_div_ql_mod_q_bdim, qlql_inv_mod_ql_div_ql_mod_q_shoup_value, qlql_inv_mod_ql_div_ql_mod_q_shoup_bdim, q_inv_mod_q_value, q_inv_mod_q_bdim, q_inv_mod_q_shoup_value, q_inv_mod_q_shoup_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor automorphism_transform_generated_plumbing(const at::Tensor & self, int64_t l, int64_t N, const at::Tensor & precomp_vec) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(precomp_vec, cur_level)) {
    return at::_ops::automorphism_transform::call(self, l, N, precomp_vec);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [precomp_vec_value, precomp_vec_bdim] = unwrapTensorAtLevel(precomp_vec, cur_level);
  auto results = batch_rule(self_value, self_bdim, l, N, precomp_vec_value, precomp_vec_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mod_raise_generated_plumbing(const at::Tensor & input, int64_t N, int64_t L0, int64_t old_prime, const at::Tensor & primes, const at::Tensor & switch_modulus_map, const at::Tensor & inverse_power_of_roots_div_two, const at::Tensor & inverse_scaled_power_of_roots_div_two, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(switch_modulus_map, cur_level) && !isBatchedAtLevel(inverse_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(inverse_scaled_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level)) {
    return at::_ops::mod_raise::call(input, N, L0, old_prime, primes, switch_modulus_map, inverse_power_of_roots_div_two, inverse_scaled_power_of_roots_div_two, power_of_roots_shoup, power_of_roots);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [switch_modulus_map_value, switch_modulus_map_bdim] = unwrapTensorAtLevel(switch_modulus_map, cur_level);
  auto [inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_power_of_roots_div_two, cur_level);
  auto [inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_scaled_power_of_roots_div_two, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto results = batch_rule(input_value, input_bdim, N, L0, old_prime, primes_value, primes_bdim, switch_modulus_map_value, switch_modulus_map_bdim, inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim, inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor extend_ciphertext_generated_plumbing(const at::Tensor & self, const at::Tensor & input, int64_t N, int64_t L0, int64_t L, int64_t composite_degree, int64_t qjProduct, const at::Tensor & qj, const at::Tensor & qhat_inv_modqj, const at::Tensor & qjProductD, const at::Tensor & qjProduct_mod, const at::Tensor & qjProduct_psinv, const at::Tensor & qj_psinv, const at::Tensor & qhat_inv_modqj_mod, const at::Tensor & qhat_inv_modqj_psinv, const at::Tensor & qjProductD_psinv, const at::Tensor & primes, const at::Tensor & inverse_power_of_roots_div_two, const at::Tensor & inverse_scaled_power_of_roots_div_two, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots, const at::Tensor & barret_ratio, const at::Tensor & barret_k) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(qj, cur_level) && !isBatchedAtLevel(qhat_inv_modqj, cur_level) && !isBatchedAtLevel(qjProductD, cur_level) && !isBatchedAtLevel(qjProduct_mod, cur_level) && !isBatchedAtLevel(qjProduct_psinv, cur_level) && !isBatchedAtLevel(qj_psinv, cur_level) && !isBatchedAtLevel(qhat_inv_modqj_mod, cur_level) && !isBatchedAtLevel(qhat_inv_modqj_psinv, cur_level) && !isBatchedAtLevel(qjProductD_psinv, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(inverse_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(inverse_scaled_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level) && !isBatchedAtLevel(barret_ratio, cur_level) && !isBatchedAtLevel(barret_k, cur_level)) {
    return at::_ops::extend_ciphertext::call(self, input, N, L0, L, composite_degree, qjProduct, qj, qhat_inv_modqj, qjProductD, qjProduct_mod, qjProduct_psinv, qj_psinv, qhat_inv_modqj_mod, qhat_inv_modqj_psinv, qjProductD_psinv, primes, inverse_power_of_roots_div_two, inverse_scaled_power_of_roots_div_two, power_of_roots_shoup, power_of_roots, barret_ratio, barret_k);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [qj_value, qj_bdim] = unwrapTensorAtLevel(qj, cur_level);
  auto [qhat_inv_modqj_value, qhat_inv_modqj_bdim] = unwrapTensorAtLevel(qhat_inv_modqj, cur_level);
  auto [qjProductD_value, qjProductD_bdim] = unwrapTensorAtLevel(qjProductD, cur_level);
  auto [qjProduct_mod_value, qjProduct_mod_bdim] = unwrapTensorAtLevel(qjProduct_mod, cur_level);
  auto [qjProduct_psinv_value, qjProduct_psinv_bdim] = unwrapTensorAtLevel(qjProduct_psinv, cur_level);
  auto [qj_psinv_value, qj_psinv_bdim] = unwrapTensorAtLevel(qj_psinv, cur_level);
  auto [qhat_inv_modqj_mod_value, qhat_inv_modqj_mod_bdim] = unwrapTensorAtLevel(qhat_inv_modqj_mod, cur_level);
  auto [qhat_inv_modqj_psinv_value, qhat_inv_modqj_psinv_bdim] = unwrapTensorAtLevel(qhat_inv_modqj_psinv, cur_level);
  auto [qjProductD_psinv_value, qjProductD_psinv_bdim] = unwrapTensorAtLevel(qjProductD_psinv, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_power_of_roots_div_two, cur_level);
  auto [inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_scaled_power_of_roots_div_two, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto [barret_ratio_value, barret_ratio_bdim] = unwrapTensorAtLevel(barret_ratio, cur_level);
  auto [barret_k_value, barret_k_bdim] = unwrapTensorAtLevel(barret_k, cur_level);
  auto results = batch_rule(self_value, self_bdim, input_value, input_bdim, N, L0, L, composite_degree, qjProduct, qj_value, qj_bdim, qhat_inv_modqj_value, qhat_inv_modqj_bdim, qjProductD_value, qjProductD_bdim, qjProduct_mod_value, qjProduct_mod_bdim, qjProduct_psinv_value, qjProduct_psinv_bdim, qj_psinv_value, qj_psinv_bdim, qhat_inv_modqj_mod_value, qhat_inv_modqj_mod_bdim, qhat_inv_modqj_psinv_value, qhat_inv_modqj_psinv_bdim, qjProductD_psinv_value, qjProductD_psinv_bdim, primes_value, primes_bdim, inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim, inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim, barret_ratio_value, barret_ratio_bdim, barret_k_value, barret_k_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_by_monomial_generated_plumbing(const at::Tensor & self, int64_t l, int64_t N, int64_t M, int64_t monomialDeg, int64_t L, const at::Tensor & primes, const at::Tensor & inverse_power_of_roots_div_two, const at::Tensor & inverse_scaled_power_of_roots_div_two, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(inverse_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(inverse_scaled_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level)) {
    return at::_ops::mul_by_monomial::call(self, l, N, M, monomialDeg, L, primes, inverse_power_of_roots_div_two, inverse_scaled_power_of_roots_div_two, power_of_roots_shoup, power_of_roots);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_power_of_roots_div_two, cur_level);
  auto [inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_scaled_power_of_roots_div_two, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto results = batch_rule(self_value, self_bdim, l, N, M, monomialDeg, L, primes_value, primes_bdim, inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim, inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & mul_by_monomial__generated_plumbing(at::Tensor & self, int64_t l, int64_t N, int64_t M, int64_t monomialDeg, int64_t L, const at::Tensor & primes, const at::Tensor & inverse_power_of_roots_div_two, const at::Tensor & inverse_scaled_power_of_roots_div_two, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(inverse_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(inverse_scaled_power_of_roots_div_two, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level)) {
    return at::_ops::mul_by_monomial_::call(self, l, N, M, monomialDeg, L, primes, inverse_power_of_roots_div_two, inverse_scaled_power_of_roots_div_two, power_of_roots_shoup, power_of_roots);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_power_of_roots_div_two, cur_level);
  auto [inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim] = unwrapTensorAtLevel(inverse_scaled_power_of_roots_div_two, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  batch_rule(self_value, self_bdim, l, N, M, monomialDeg, L, primes_value, primes_bdim, inverse_power_of_roots_div_two_value, inverse_power_of_roots_div_two_bdim, inverse_scaled_power_of_roots_div_two_value, inverse_scaled_power_of_roots_div_two_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor encode_generated_plumbing(const at::Tensor & input, int64_t N, int64_t cur_limbs, int64_t slots, double scaling_factor, bool is_ext, int64_t sizeP, const at::Tensor & primes, const at::Tensor & max_int_diffs, const at::Tensor & barret_ratio, const at::Tensor & barret_k, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(max_int_diffs, cur_level) && !isBatchedAtLevel(barret_ratio, cur_level) && !isBatchedAtLevel(barret_k, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level)) {
    return at::_ops::encode::call(input, N, cur_limbs, slots, scaling_factor, is_ext, sizeP, primes, max_int_diffs, barret_ratio, barret_k, power_of_roots_shoup, power_of_roots);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [max_int_diffs_value, max_int_diffs_bdim] = unwrapTensorAtLevel(max_int_diffs, cur_level);
  auto [barret_ratio_value, barret_ratio_bdim] = unwrapTensorAtLevel(barret_ratio, cur_level);
  auto [barret_k_value, barret_k_bdim] = unwrapTensorAtLevel(barret_k, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto results = batch_rule(input_value, input_bdim, N, cur_limbs, slots, scaling_factor, is_ext, sizeP, primes_value, primes_bdim, max_int_diffs_value, max_int_diffs_bdim, barret_ratio_value, barret_ratio_bdim, barret_k_value, barret_k_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> encrypt_generated_plumbing(const at::Tensor & ptx, const at::Tensor & pk0, const at::Tensor & pk1, int64_t l, int64_t logn, int64_t nh, const at::Tensor & moduliP_scalar, const at::Tensor & moduliQ_scalar, const at::Tensor & primes, const at::Tensor & max_int_diffs, const at::Tensor & barret_ratio, const at::Tensor & barret_k, const at::Tensor & power_of_roots_shoup, const at::Tensor & power_of_roots) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(ptx, cur_level) && !isBatchedAtLevel(pk0, cur_level) && !isBatchedAtLevel(pk1, cur_level) && !isBatchedAtLevel(moduliP_scalar, cur_level) && !isBatchedAtLevel(moduliQ_scalar, cur_level) && !isBatchedAtLevel(primes, cur_level) && !isBatchedAtLevel(max_int_diffs, cur_level) && !isBatchedAtLevel(barret_ratio, cur_level) && !isBatchedAtLevel(barret_k, cur_level) && !isBatchedAtLevel(power_of_roots_shoup, cur_level) && !isBatchedAtLevel(power_of_roots, cur_level)) {
    return at::_ops::encrypt::call(ptx, pk0, pk1, l, logn, nh, moduliP_scalar, moduliQ_scalar, primes, max_int_diffs, barret_ratio, barret_k, power_of_roots_shoup, power_of_roots);
  }
  auto [ptx_value, ptx_bdim] = unwrapTensorAtLevel(ptx, cur_level);
  auto [pk0_value, pk0_bdim] = unwrapTensorAtLevel(pk0, cur_level);
  auto [pk1_value, pk1_bdim] = unwrapTensorAtLevel(pk1, cur_level);
  auto [moduliP_scalar_value, moduliP_scalar_bdim] = unwrapTensorAtLevel(moduliP_scalar, cur_level);
  auto [moduliQ_scalar_value, moduliQ_scalar_bdim] = unwrapTensorAtLevel(moduliQ_scalar, cur_level);
  auto [primes_value, primes_bdim] = unwrapTensorAtLevel(primes, cur_level);
  auto [max_int_diffs_value, max_int_diffs_bdim] = unwrapTensorAtLevel(max_int_diffs, cur_level);
  auto [barret_ratio_value, barret_ratio_bdim] = unwrapTensorAtLevel(barret_ratio, cur_level);
  auto [barret_k_value, barret_k_bdim] = unwrapTensorAtLevel(barret_k, cur_level);
  auto [power_of_roots_shoup_value, power_of_roots_shoup_bdim] = unwrapTensorAtLevel(power_of_roots_shoup, cur_level);
  auto [power_of_roots_value, power_of_roots_bdim] = unwrapTensorAtLevel(power_of_roots, cur_level);
  auto results = batch_rule(ptx_value, ptx_bdim, pk0_value, pk0_bdim, pk1_value, pk1_bdim, l, logn, nh, moduliP_scalar_value, moduliP_scalar_bdim, moduliQ_scalar_value, moduliQ_scalar_bdim, primes_value, primes_bdim, max_int_diffs_value, max_int_diffs_bdim, barret_ratio_value, barret_ratio_bdim, barret_k_value, barret_k_bdim, power_of_roots_shoup_value, power_of_roots_shoup_bdim, power_of_roots_value, power_of_roots_bdim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pre_encode_generated_plumbing(const at::Tensor & input, int64_t slots, int64_t M, const at::Tensor & rotGroup, const at::Tensor & ksiPows, const at::Tensor & bitrev) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(rotGroup, cur_level) && !isBatchedAtLevel(ksiPows, cur_level) && !isBatchedAtLevel(bitrev, cur_level)) {
    return at::_ops::pre_encode::call(input, slots, M, rotGroup, ksiPows, bitrev);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [rotGroup_value, rotGroup_bdim] = unwrapTensorAtLevel(rotGroup, cur_level);
  auto [ksiPows_value, ksiPows_bdim] = unwrapTensorAtLevel(ksiPows, cur_level);
  auto [bitrev_value, bitrev_bdim] = unwrapTensorAtLevel(bitrev, cur_level);
  auto results = batch_rule(input_value, input_bdim, slots, M, rotGroup_value, rotGroup_bdim, ksiPows_value, ksiPows_bdim, bitrev_value, bitrev_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor batched_pairwise_mac_generated_plumbing(const at::Tensor & cipher, const at::Tensor & plaintext, const at::Tensor & param_primes, const at::Tensor & barret_ratio, const at::Tensor & barret_k, int64_t num_batches, int64_t num_cipher, int64_t curr_limbs, int64_t N) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(cipher, cur_level) && !isBatchedAtLevel(plaintext, cur_level) && !isBatchedAtLevel(param_primes, cur_level) && !isBatchedAtLevel(barret_ratio, cur_level) && !isBatchedAtLevel(barret_k, cur_level)) {
    return at::_ops::batched_pairwise_mac::call(cipher, plaintext, param_primes, barret_ratio, barret_k, num_batches, num_cipher, curr_limbs, N);
  }
  auto [cipher_value, cipher_bdim] = unwrapTensorAtLevel(cipher, cur_level);
  auto [plaintext_value, plaintext_bdim] = unwrapTensorAtLevel(plaintext, cur_level);
  auto [param_primes_value, param_primes_bdim] = unwrapTensorAtLevel(param_primes, cur_level);
  auto [barret_ratio_value, barret_ratio_bdim] = unwrapTensorAtLevel(barret_ratio, cur_level);
  auto [barret_k_value, barret_k_bdim] = unwrapTensorAtLevel(barret_k, cur_level);
  auto results = batch_rule(cipher_value, cipher_bdim, plaintext_value, plaintext_bdim, param_primes_value, param_primes_bdim, barret_ratio_value, barret_ratio_bdim, barret_k_value, barret_k_bdim, num_batches, num_cipher, curr_limbs, N);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cpmul_broadcast_pt_generated_plumbing(const at::Tensor & cipher, const at::Tensor & plaintext, const at::Tensor & param_primes, const at::Tensor & barret_mu, int64_t num_cipher, int64_t curr_limbs, int64_t N) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(cipher, cur_level) && !isBatchedAtLevel(plaintext, cur_level) && !isBatchedAtLevel(param_primes, cur_level) && !isBatchedAtLevel(barret_mu, cur_level)) {
    return at::_ops::cpmul_broadcast_pt::call(cipher, plaintext, param_primes, barret_mu, num_cipher, curr_limbs, N);
  }
  auto [cipher_value, cipher_bdim] = unwrapTensorAtLevel(cipher, cur_level);
  auto [plaintext_value, plaintext_bdim] = unwrapTensorAtLevel(plaintext, cur_level);
  auto [param_primes_value, param_primes_bdim] = unwrapTensorAtLevel(param_primes, cur_level);
  auto [barret_mu_value, barret_mu_bdim] = unwrapTensorAtLevel(barret_mu, cur_level);
  auto results = batch_rule(cipher_value, cipher_bdim, plaintext_value, plaintext_bdim, param_primes_value, param_primes_bdim, barret_mu_value, barret_mu_bdim, num_cipher, curr_limbs, N);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hardtanh_generated_plumbing(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardtanh::call(self, min_val, max_val);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min_val, max_val);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & hardtanh__generated_plumbing(at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardtanh_::call(self, min_val, max_val);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, min_val, max_val);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor leaky_relu_generated_plumbing(const at::Tensor & self, const at::Scalar & negative_slope) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::leaky_relu::call(self, negative_slope);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, negative_slope);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & leaky_relu__generated_plumbing(at::Tensor & self, const at::Scalar & negative_slope) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::leaky_relu_::call(self, negative_slope);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, negative_slope);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor leaky_relu_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::leaky_relu_backward::call(grad_output, self, negative_slope, self_is_result);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, negative_slope, self_is_result);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor elu_generated_plumbing(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::elu::call(self, alpha, scale, input_scale);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, alpha, scale, input_scale);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & elu__generated_plumbing(at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::elu_::call(self, alpha, scale, input_scale);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, alpha, scale, input_scale);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor elu_backward_generated_plumbing(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self_or_result, cur_level)) {
    return at::_ops::elu_backward::call(grad_output, alpha, scale, input_scale, is_result, self_or_result);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto [self_or_result_value, self_or_result_bdim] = unwrapTensorAtLevel(self_or_result, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, alpha, scale, input_scale, is_result, self_or_result_value, self_or_result_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _prelu_kernel_generated_plumbing(const at::Tensor & self, const at::Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::_prelu_kernel::call(self, weight);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [weight_value, weight_bdim] = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> _prelu_kernel_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::_prelu_kernel_backward::call(grad_output, self, weight);
  }
  auto [grad_output_value, grad_output_bdim] = unwrapTensorAtLevel(grad_output, cur_level);
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [weight_value, weight_bdim] = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, weight_value, weight_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log_sigmoid_forward::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _sparse_compressed_tensor_unsafe_generated_plumbing(const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(compressed_indices, cur_level) && !isBatchedAtLevel(plain_indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::_sparse_compressed_tensor_unsafe::call(compressed_indices, plain_indices, values, size, dtype, layout, device, pin_memory);
  }
  auto [compressed_indices_value, compressed_indices_bdim] = unwrapTensorAtLevel(compressed_indices, cur_level);
  auto [plain_indices_value, plain_indices_bdim] = unwrapTensorAtLevel(plain_indices, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(compressed_indices_value, compressed_indices_bdim, plain_indices_value, plain_indices_bdim, values_value, values_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
const at::Tensor & resize_as_sparse__generated_plumbing(const at::Tensor & self, const at::Tensor & the_template) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(the_template, cur_level)) {
    return at::_ops::resize_as_sparse_::call(self, the_template);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [the_template_value, the_template_bdim] = unwrapTensorAtLevel(the_template, cur_level);
  batch_rule(self_value, self_bdim, the_template_value, the_template_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _fft_c2c_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef dim, int64_t normalization, bool forward) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_fft_c2c::call(self, dim, normalization, forward);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, normalization, forward);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor linalg_vector_norm_generated_plumbing(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::linalg_vector_norm::call(self, ord, dim, keepdim, dtype);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, ord, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_generated_plumbing(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(offsets, cur_level) && !isBatchedAtLevel(per_sample_weights, cur_level)) {
    return at::_ops::_embedding_bag::call(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  }
  auto [weight_value, weight_bdim] = unwrapTensorAtLevel(weight, cur_level);
  auto [indices_value, indices_bdim] = unwrapTensorAtLevel(indices, cur_level);
  auto [offsets_value, offsets_bdim] = unwrapTensorAtLevel(offsets, cur_level);
  std::optional<Tensor> per_sample_weights_value;
  std::optional<int64_t> per_sample_weights_bdim;
  if (per_sample_weights) {
      std::tie(per_sample_weights_value, per_sample_weights_bdim) = unwrapTensorAtLevel(per_sample_weights.value(), cur_level);
  }
  auto results = batch_rule(weight_value, weight_bdim, indices_value, indices_bdim, offsets_value, offsets_bdim, scale_grad_by_freq, mode, sparse, per_sample_weights_value, per_sample_weights_bdim, include_last_offset, padding_idx);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fbgemm_linear_fp16_weight_fp32_activation_generated_plumbing(const at::Tensor & input, const at::Tensor & packed_weight, const ::std::optional<at::Tensor> & bias) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(packed_weight, cur_level) && !isBatchedAtLevel(bias, cur_level)) {
    return at::_ops::fbgemm_linear_fp16_weight_fp32_activation::call(input, packed_weight, bias);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [packed_weight_value, packed_weight_bdim] = unwrapTensorAtLevel(packed_weight, cur_level);
  std::optional<Tensor> bias_value;
  std::optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, packed_weight_value, packed_weight_bdim, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fbgemm_pack_gemm_matrix_fp16_generated_plumbing(const at::Tensor & input) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::fbgemm_pack_gemm_matrix_fp16::call(input);
  }
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _wrapped_linear_prepack_generated_plumbing(const at::Tensor & weight, const at::Tensor & weight_scale, const at::Tensor & weight_zero_point, const at::Tensor & bias) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(weight_scale, cur_level) && !isBatchedAtLevel(weight_zero_point, cur_level) && !isBatchedAtLevel(bias, cur_level)) {
    return at::_ops::_wrapped_linear_prepack::call(weight, weight_scale, weight_zero_point, bias);
  }
  auto [weight_value, weight_bdim] = unwrapTensorAtLevel(weight, cur_level);
  auto [weight_scale_value, weight_scale_bdim] = unwrapTensorAtLevel(weight_scale, cur_level);
  auto [weight_zero_point_value, weight_zero_point_bdim] = unwrapTensorAtLevel(weight_zero_point, cur_level);
  auto [bias_value, bias_bdim] = unwrapTensorAtLevel(bias, cur_level);
  auto results = batch_rule(weight_value, weight_bdim, weight_scale_value, weight_scale_bdim, weight_zero_point_value, weight_zero_point_bdim, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> _histogramdd_bin_edges_generated_plumbing(const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::_histogramdd_bin_edges::call(self, bins, range, weight, density);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins, range, weight_value, weight_bdim, density);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _histogramdd_from_bin_cts_generated_plumbing(const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::_histogramdd_from_bin_cts::call(self, bins, range, weight, density);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins, range, weight_value, weight_bdim, density);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _histogramdd_from_bin_tensors_generated_plumbing(const at::Tensor & self, at::TensorList bins, const ::std::optional<at::Tensor> & weight, bool density) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(bins, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::_histogramdd_from_bin_tensors::call(self, bins, weight, density);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins, weight_value, weight_bdim, density);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,::std::vector<at::Tensor>> histogramdd_generated_plumbing(const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::histogramdd::call(self, bins, range, weight, density);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins, range, weight_value, weight_bdim, density);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatchedVector(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bernoulli_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(p, cur_level)) {
    return at::_ops::bernoulli_Tensor::call(self, p, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [p_value, p_bdim] = unwrapTensorAtLevel(p, cur_level);
  auto results = batch_rule(self_value, self_bdim, p_value, p_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor resize_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::resize::call(self, size, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _resize_output_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef size, at::Device device) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_resize_output::call(self, size, device);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, device);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _index_put_impl_generated_plumbing(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::_index_put_impl::call(self, indices, values, accumulate, unsafe);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices, values_value, values_bdim, accumulate, unsafe);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void unsafe_split_Tensor_out_generated_plumbing(const at::Tensor & self, c10::SymInt split_size, int64_t dim, at::TensorList out) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(out, cur_level)) {
    return at::_ops::unsafe_split_Tensor_out::call(self, split_size, dim, out);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, split_size, dim, out);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void unsafe_split_with_sizes_out_generated_plumbing(const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim, at::TensorList out) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(out, cur_level)) {
    return at::_ops::unsafe_split_with_sizes_out::call(self, split_sizes, dim, out);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, split_sizes, dim, out);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor resize_as_generated_plumbing(const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(the_template, cur_level)) {
    return at::_ops::resize_as::call(self, the_template, memory_format);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [the_template_value, the_template_bdim] = unwrapTensorAtLevel(the_template, cur_level);
  auto results = batch_rule(self_value, self_bdim, the_template_value, the_template_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor zero_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::zero::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _coalesced_generated_plumbing(const at::Tensor & self, bool coalesced) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_coalesced::call(self, coalesced);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, coalesced);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor set_source_Storage_generated_plumbing(const at::Tensor & self, at::Storage source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::set_source_Storage::call(self, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor set_source_Storage_storage_offset_generated_plumbing(const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::set_source_Storage_storage_offset::call(self, source, storage_offset, size, stride);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source, storage_offset, size, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor set_source_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & source) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::set_source_Tensor::call(self, source);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [source_value, source_bdim] = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, source_value, source_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor set_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::set::call(self);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor random_from_generated_plumbing(const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::random_from::call(self, from, to, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, from, to, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor random_to_generated_plumbing(const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::random_to::call(self, to, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, to, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor random_generated_plumbing(const at::Tensor & self, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::random::call(self, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor uniform_generated_plumbing(const at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::uniform::call(self, from, to, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, from, to, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor exponential_generated_plumbing(const at::Tensor & self, double lambd, ::std::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exponential::call(self, lambd, generator);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, lambd, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor resize_as_sparse_generated_plumbing(const at::Tensor & self, const at::Tensor & the_template) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(the_template, cur_level)) {
    return at::_ops::resize_as_sparse::call(self, the_template);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [the_template_value, the_template_bdim] = unwrapTensorAtLevel(the_template, cur_level);
  auto results = batch_rule(self_value, self_bdim, the_template_value, the_template_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
void _histogramdd_bin_edges_out_generated_plumbing(const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density, at::TensorList out) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(out, cur_level)) {
    return at::_ops::_histogramdd_bin_edges_out::call(self, bins, range, weight, density, out);
  }
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  std::optional<Tensor> weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  batch_rule(self_value, self_bdim, bins, range, weight_value, weight_bdim, density, out);
}

}} // namespace at::functorch
