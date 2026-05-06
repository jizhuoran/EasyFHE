
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at::functorch {

#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
}

static void unsupportedData(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false, "mutating directly with `.data` under vmap transform is not allowed.");
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatchedDecomposition, m) {
  OP_DECOMPOSE2(__and__, Scalar);
  OP_DECOMPOSE2(__and__, Tensor);
  OP_DECOMPOSE2(__iand__, Tensor);
  OP_DECOMPOSE2(__iand__, Scalar);
  OP_DECOMPOSE2(__ior__, Tensor);
  OP_DECOMPOSE2(__ior__, Scalar);
  OP_DECOMPOSE2(__ixor__, Tensor);
  OP_DECOMPOSE2(__ixor__, Scalar);
  OP_DECOMPOSE2(__or__, Tensor);
  OP_DECOMPOSE2(__or__, Scalar);
  OP_DECOMPOSE2(__xor__, Tensor);
  OP_DECOMPOSE2(__xor__, Scalar);
  OP_DECOMPOSE(absolute);
  OP_DECOMPOSE(absolute_);
  OP_DECOMPOSE(arctan2);
  OP_DECOMPOSE(arctan2_);
  OP_DECOMPOSE(argsort);
  OP_DECOMPOSE2(argsort, stable);
  OP_DECOMPOSE(adjoint);
  OP_DECOMPOSE(arccos);
  OP_DECOMPOSE(arccos_);
  OP_DECOMPOSE(arccosh);
  OP_DECOMPOSE(arccosh_);
  OP_DECOMPOSE(arcsin);
  OP_DECOMPOSE(arcsin_);
  OP_DECOMPOSE(arcsinh);
  OP_DECOMPOSE(arcsinh_);
  OP_DECOMPOSE(arctan);
  OP_DECOMPOSE(arctan_);
  OP_DECOMPOSE(arctanh);
  OP_DECOMPOSE(arctanh_);
  OP_DECOMPOSE(atleast_1d);
  OP_DECOMPOSE2(atleast_1d, Sequence);
  OP_DECOMPOSE(atleast_2d);
  OP_DECOMPOSE2(atleast_2d, Sequence);
  OP_DECOMPOSE(atleast_3d);
  OP_DECOMPOSE2(atleast_3d, Sequence);
  OP_DECOMPOSE(broadcast_tensors);
  // m.impl("broadcast_to", native::broadcast_to_symint); // removed for EasyFHE
  OP_DECOMPOSE(chunk);
  OP_DECOMPOSE(clip);
  OP_DECOMPOSE2(clip, Tensor );
  OP_DECOMPOSE(concat);
  OP_DECOMPOSE(conj_physical);
  OP_DECOMPOSE(contiguous);
  OP_DECOMPOSE2(dsplit, int);
  OP_DECOMPOSE2(dsplit, array);
  OP_DECOMPOSE(diff);
  OP_DECOMPOSE(diag);
  OP_DECOMPOSE(dstack);
  OP_DECOMPOSE(expand_as);
  OP_DECOMPOSE(fix);
  OP_DECOMPOSE(fliplr);
  OP_DECOMPOSE(flipud);
  OP_DECOMPOSE2(flatten, using_ints);
  OP_DECOMPOSE2(float_power, Tensor_Tensor);
  OP_DECOMPOSE2(float_power, Tensor_Scalar);
  OP_DECOMPOSE2(float_power, Scalar);
  OP_DECOMPOSE(gather_backward);
  OP_DECOMPOSE2(gradient, scalarint);
  OP_DECOMPOSE2(gradient, scalararray);
  OP_DECOMPOSE2(gradient, array);
  OP_DECOMPOSE2(gradient, scalarrayint);
  OP_DECOMPOSE2(gradient, scalarrayarray);
  OP_DECOMPOSE2(gradient, tensorarrayint);
  OP_DECOMPOSE2(gradient, tensorarray);
  OP_DECOMPOSE2(greater_equal, Tensor );
  OP_DECOMPOSE2(greater_equal, Scalar );
  OP_DECOMPOSE2(greater, Tensor );
  OP_DECOMPOSE2(hsplit, int);
  OP_DECOMPOSE2(hsplit, array);
  OP_DECOMPOSE(hstack);
  // m.impl("index_select_backward", native::index_select_backward_symint); // removed for EasyFHE
  OP_DECOMPOSE(isfinite);
  OP_DECOMPOSE(isreal);
  OP_DECOMPOSE(concatenate);
  OP_DECOMPOSE2(less_equal, Tensor );
  OP_DECOMPOSE2(less, Tensor );
  OP_DECOMPOSE(cumprod_backward);
  OP_DECOMPOSE(matrix_H);
  OP_DECOMPOSE2(max, other );
  OP_DECOMPOSE(meshgrid);
  OP_DECOMPOSE2(meshgrid, indexing);
  OP_DECOMPOSE(mH);
  OP_DECOMPOSE2(min, other );
  OP_DECOMPOSE2(moveaxis, intlist);
  OP_DECOMPOSE2(movedim, int);
  OP_DECOMPOSE2(movedim, intlist);
  OP_DECOMPOSE(msort);
  OP_DECOMPOSE(mT);
  OP_DECOMPOSE(nanmean);
  // m.impl("narrow", native::narrow_symint); // removed for EasyFHE
  OP_DECOMPOSE(negative);
  OP_DECOMPOSE2(not_equal, Tensor );
  OP_DECOMPOSE(positive);
  OP_DECOMPOSE(ravel);
  // removed for EasyFHE: repeat_interleave.self_int
  // removed for EasyFHE: repeat_interleave.self_Tensor
  // m.impl("reshape", native::reshape_symint); // removed for EasyFHE
  OP_DECOMPOSE(resolve_conj);
  OP_DECOMPOSE(resolve_neg);
  OP_DECOMPOSE(row_stack);


  // m.impl("split.sizes", native::split_symint); // removed for EasyFHE
  OP_DECOMPOSE(square);
  OP_DECOMPOSE(numpy_T);
  OP_DECOMPOSE(reshape_as);
  OP_DECOMPOSE2(result_type, Tensor);
  OP_DECOMPOSE2(result_type, Scalar);
  OP_DECOMPOSE2(result_type, Scalar_Tensor);
  OP_DECOMPOSE2(result_type, Scalar_Scalar);
  OP_DECOMPOSE(is_same_size);
  OP_DECOMPOSE(view_as);
  OP_DECOMPOSE2(size, int);
  OP_DECOMPOSE(is_complex);
  OP_DECOMPOSE(std);
  OP_DECOMPOSE2(std, dim);
  OP_DECOMPOSE(std_mean);
  OP_DECOMPOSE2(std_mean, dim);
  OP_DECOMPOSE(swapaxes);
  OP_DECOMPOSE2(subtract, Tensor);
  // m.impl("sum_to_size", native::sum_to_size_symint); // removed for EasyFHE
  OP_DECOMPOSE(swapdims);
  OP_DECOMPOSE(take_along_dim);
  // m.impl("tensor_split.indices", native::tensor_split_indices_symint); // removed for EasyFHE
  // m.impl("tensor_split.sections", native::tensor_split_sections_symint); // removed for EasyFHE
  // m.impl("tile", native::tile_symint); // removed for EasyFHE
  OP_DECOMPOSE(unsafe_chunk);
  // removed for EasyFHE: value_selecting_reduction_backward
  OP_DECOMPOSE(var);
  OP_DECOMPOSE2(var, dim);
  OP_DECOMPOSE(var_mean);
  OP_DECOMPOSE2(var_mean, dim);
  OP_DECOMPOSE2(vsplit, int);
  OP_DECOMPOSE2(vsplit, array);
  OP_DECOMPOSE(vstack);
  OP_DECOMPOSE2(where, ScalarOther);
  OP_DECOMPOSE2(where, ScalarSelf);
  OP_DECOMPOSE2(where, Scalar);
  // m.impl("unflatten.int", native::unflatten_symint); // removed for EasyFHE
  OP_DECOMPOSE(type_as);
  OP_DECOMPOSE(diagonal_copy);
  OP_DECOMPOSE(alias_copy);
  m.impl("as_strided_copy", native::as_strided_copy_symint);
  OP_DECOMPOSE(swapdims_);
  OP_DECOMPOSE(swapaxes_);
  OP_DECOMPOSE(unfold_copy);
  // Easy way to decompose upsample*.vec overloads instead of introducing *_symint methods
  // if used OP_DECOMPOSE2.

  // views on complex tensor
  OP_DECOMPOSE(imag);
  OP_DECOMPOSE(real);

  // divide, alias for div
  OP_DECOMPOSE2(divide, Tensor);
  OP_DECOMPOSE2(divide_, Tensor);
  OP_DECOMPOSE2(divide, Scalar);
  OP_DECOMPOSE2(divide, Tensor_mode);
  OP_DECOMPOSE2(divide_, Tensor_mode);
  OP_DECOMPOSE2(divide, Scalar_mode);
  OP_DECOMPOSE2(divide_, Scalar_mode);

  // divide, alias for div
  OP_DECOMPOSE2(true_divide, Tensor);
  OP_DECOMPOSE2(true_divide_, Tensor);
  OP_DECOMPOSE2(true_divide, Scalar);
  OP_DECOMPOSE2(true_divide_, Scalar);

  // multiply, alias for mul
  OP_DECOMPOSE2(multiply, Tensor)
  OP_DECOMPOSE2(multiply_, Tensor)
  OP_DECOMPOSE2(multiply, Scalar)
  OP_DECOMPOSE2(multiply_, Scalar)


  // comparison ops
  OP_DECOMPOSE2(greater, Scalar);
  OP_DECOMPOSE2(less_equal, Scalar);
  OP_DECOMPOSE2(less, Scalar);
  OP_DECOMPOSE2(not_equal, Scalar);
  m.impl("_has_compatible_shallow_copy_type", torch::CppFunction::makeFromBoxedFunction<&unsupportedData>());

  // to.*
  OP_DECOMPOSE2(to, device);
  OP_DECOMPOSE2(to, dtype);
  OP_DECOMPOSE2(to, dtype_layout);
  OP_DECOMPOSE2(to, other);

  // Random ops that are also registered here
}

} // namespace at::functorch
