# Be extra careful when you edit this file, because it affects AOTInductor ABI compatibility. See
# https://github.com/pytorch/pytorch/blob/7e86a7c0155295539996e0cf422883571126073e/torchgen/gen.py#L2424-L2436
# for details.
#
# The inductor_fallback_ops list is based on the fallback ops from torch/_inductor/lowering.py.
#
# Generally speaking, it is ok to add a new op to the list, but you need to run
# `python torchgen/gen.py --update-aoti-c-shim` in order to regenerate C shim header files.
# But it is NOT ok to remove an existing fallback op from the list, since that will break
# some existing AOTInductor-compiled models.
#
# A fallback op version defaults to 1. If you want to extend an existing fallback op by adding
# a new argument with a default value, while it is fine in the Python world, it will be BC-breaking
# when generating C shim. Thus you need to bump up the version number of that fallback op by
# updating the entry in the inductor_fallback_ops list, adding a new version number with a list
# of new arguments, and then run `python torchgen/gen.py --update-aoti-c-shim` to regenerate.
#
# A "since" key (top-level for v1, or nested inside a vN entry of the form
#       {"new_args": [...], "since": "TORCH_VERSION_X_Y_Z"})
# records the earliest TORCH_VERSION at which that op variant became available. When present,
# codegen wraps the generated declaration in a
#   #if TORCH_FEATURE_VERSION >= <since>
#   ...
#   #endif
# guard. When absent, the declaration is emitted ungated, which would signify that the op was
# available since torch 2.9. ALL NEWLY ADDED OPS MUST INCLUDE A "since" KEY WITH THE VALUE SET
# TO THE CURRENT TORCH VERSION (the version when the op was added).

inductor_fallback_ops: dict[str, dict[str, str | dict[str, list[str] | str]]] = {
    "aten._efficientzerotensor.default": {},
    "aten._histogramdd_from_bin_cts.default": {},
    "aten._segment_reduce_backward.default": {},
    "aten.abs.default": {},
    "aten.add.Scalar": {},
    "aten.add.Tensor": {},
    "aten.angle.default": {},
    "aten.bernoulli_.float": {},
    "aten.bernoulli_.Tensor": {},
    "aten.bucketize.Tensor": {},
    "aten.cat.default": {},
    "aten.cummax.default": {},
    "aten.cummin.default": {},
    "aten.cumprod.default": {},
    "aten.cumsum.default": {},
    "aten.exponential.default": {},
    "aten.fill_.Scalar": {},
    "aten.gcd.default": {},
    "aten.histc.default": {},
    "aten.histogram.bin_ct": {},
    "aten.index_put.default": {},
    "aten.index_reduce.default": {},
    "aten.index.Tensor": {},
    "aten.kthvalue.default": {},
    "aten.logcumsumexp.default": {},
    "aten.masked_scatter_backward.default": {},
    "aten.masked_scatter.default": {},
    "aten.masked_select.default": {},
    "aten.median.default": {},
    "aten.mode.default": {},
    "aten.mul.Scalar": {},
    "aten.mul.Tensor": {},
    "aten.nanmedian.default": {},
    "aten.narrow.default": {},
    "aten.nonzero.default": {},
    "aten.nonzero_static.default": {"since": "TORCH_VERSION_2_11_0"},
    "aten.normal_functional.default": {},
    "aten.permute.default": {},
    "aten.polar.default": {},
    "aten.pow.Scalar": {},
    "aten.pow.Tensor_Scalar": {},
    "aten.pow.Tensor_Tensor": {},
    "aten.rand.default": {},
    "aten.rand.generator": {},
    "aten.randint.default": {},
    "aten.randint.generator": {},
    "aten.randint.low_out": {},
    "aten.randint.low": {},
    "aten.randn.default": {},
    "aten.randn.generator": {},
    "aten.randperm.default": {},
    "aten.rand_like.default": {"since": "TORCH_VERSION_2_12_0"},
    "aten.rand_like.generator": {"since": "TORCH_VERSION_2_12_0"},
    "aten.randint_like.default": {"since": "TORCH_VERSION_2_12_0"},
    "aten.randint_like.low_dtype": {"since": "TORCH_VERSION_2_12_0"},
    "aten.randn_like.default": {"since": "TORCH_VERSION_2_12_0"},
    "aten.randn_like.generator": {"since": "TORCH_VERSION_2_12_0"},
    "aten.repeat_interleave.Tensor": {},
    "aten.reshape.default": {},
    "aten.resize_.default": {},
    "aten.resize_as_.default": {},
    "aten.scatter_reduce.two_out": {},
    "aten.scatter.src_out": {},
    "aten.scatter.value_out": {},
    "aten.searchsorted.Scalar": {},
    "aten.searchsorted.Tensor": {},
    "aten.segment_reduce.default": {},
    "aten.set_.source_Tensor": {},
    "aten.slice.Tensor": {},
    "aten.sort.default": {},
    "aten.sort.stable": {},
    "aten.squeeze.dim": {},
    "aten.topk.default": {},
    "aten.uniform.default": {},
    "aten.view_as_complex.default": {},
    "aten.view_as_real.default": {},
    "aten.view.dtype": {},
}

# `python torchgen/gen.py --update-aoti-c-shim` will automatically generate
# c_shim_aten.{h/cpp} based on the list below.
# Operators in this list are intended to be used in torch/csrc/stable/ops.h
# Unlike other c_shims, operators in this file do not bypass the dispatcher.
# The same BC rules apply as inductor_fallback_ops, read about the "since"
# key above.
aten_shimified_ops: dict[str, dict[str, str | dict[str, list[str] | str]]] = {
    "aten.fill_.Scalar": {},
    "aten.narrow.default": {},
    "aten.amax.default": {},
    "aten.new_empty.default": {},
    "aten.new_zeros.default": {},
    "aten.full.default": {"since": "TORCH_VERSION_2_10_0"},
    "aten.subtract.Tensor": {"since": "TORCH_VERSION_2_10_0"},
}
