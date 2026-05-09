#!/usr/bin/env python3
"""Classify ops in native_functions.yaml as KEEP or DELETE for EasyFHE."""

import re
import sys
from pathlib import Path

YAML_PATH = Path(__file__).parent.parent / "aten/src/ATen/native/native_functions.yaml"

# --- KEEP patterns ---
# Strategy: define what we KEEP, delete everything else.

KEEP_REDUCTION = [
    "sum", "mean", "median", "mode", "std", "var", "norm", "cumsum", "cumprod",
    "cummax", "cummin", "prod", "all", "any", "argmax", "argmin", "max", "min",
    "amax", "amin", "aminmax", "_aminmax", "nansum", "nanmean", "nanmedian", "count_nonzero",
    "logsumexp", "logcumsumexp", "_logcumsumexp", "_cummin_helper", "_cummax_helper",
    "_is_all_true", "_is_any_true",
    "native_norm", "quantile", "nanquantile",
    "cummaxmin_backward", "value_selecting_reduction_backward",
    "corrcoef", "cov",
]

KEEP_INDEXING = [
    "index", "slice", "select", "gather", "scatter", "put", "take", "take_along_dim",
    "index_put", "index_copy", "index_add", "index_fill", "index_reduce",
    "index_select", "masked_fill", "masked_scatter", "masked_select",
    "where", "nonzero", "narrow", "narrow_copy", "diagonal", "diag",
    "tril", "triu", "trace", "flip", "fliplr", "flipud", "roll",
    "rot90", "bucketize", "searchsorted", "unique", "unique_consecutive",
    "_unique", "_unique2", "sort", "argsort", "topk", "kthvalue",
    "msort", "renorm",
]

KEEP_ARITHMETIC = [
    "add", "sub", "mul", "div", "pow", "sqrt", "rsqrt", "square",
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "abs", "neg", "negative", "positive", "sgn", "sign", "signbit",
    "ceil", "floor", "round", "trunc", "frac", "fix",
    "reciprocal", "remainder", "fmod", "gcd", "lcm",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "hypot", "sinc", "deg2rad", "rad2deg",
    "clamp", "clip", "lerp", "addcdiv", "addcmul",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_left_shift", "bitwise_right_shift",
    "logical_and", "logical_or", "logical_xor", "logical_not",
    "conj", "conj_physical", "_conj", "_conj_physical",
    "real", "imag", "_neg_view", "resolve_conj", "resolve_neg",
    "copysign", "nextafter", "heaviside", "frexp", "ldexp",
    "float_power", "true_divide", "floor_divide",
    "multiply", "subtract", "divide",  # aliases
    "sigmoid", "sigmoid_", "sigmoid_backward",
    "erf", "erf_",
    "logit", "logit_", "logit_backward",
    "logaddexp", "logaddexp2",
    "rsub",
    "isposinf", "isneginf",
    "angle",
    "igamma", "igamma_", "igammac", "igammac_",
    # aliases
    "absolute", "absolute_",
    "arccos", "arccos_", "arccosh", "arccosh_",
    "arcsin", "arcsin_", "arcsinh", "arcsinh_",
    "arctan", "arctan_", "arctan2", "arctan2_", "arctanh", "arctanh_",
    # bitwise operators
    "__and__", "__or__", "__xor__", "__lshift__", "__rshift__",
    "__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__",
]

KEEP_COMPARISON = [
    "eq", "ne", "lt", "le", "gt", "ge",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "isinf", "isnan", "isfinite", "isreal", "isclose", "allclose",
    "maximum", "minimum", "fmax", "fmin",
]

KEEP_SHAPE = [
    "view", "reshape", "_reshape_alias", "transpose", "permute", "movedim", "moveaxis",
    "squeeze", "unsqueeze", "flatten", "unflatten",
    "expand", "expand_as", "expand_copy", "broadcast_to", "broadcast_tensors",
    "contiguous", "cat", "concat", "concatenate", "_cat",
    "stack", "hstack", "vstack", "dstack", "column_stack", "row_stack",
    "split", "split_with_sizes", "split_copy", "split_with_sizes_copy",
    "unsafe_split", "unsafe_split_with_sizes",
    "tensor_split", "chunk", "unsafe_chunk",
    "repeat", "repeat_interleave", "tile",
    "as_strided", "as_strided_", "as_strided_copy", "as_strided_scatter",
    "t", "t_", "t_copy", "T", "H", "mT", "mH",
    "swapaxes", "swapdims",
    "narrow", "narrow_copy",
    "slice_scatter", "select_scatter", "diagonal_scatter",
    "alias", "alias_copy",
    "_reshape_alias_copy",
    "atleast_1d", "atleast_2d", "atleast_3d",
    "hsplit", "vsplit", "dsplit",
    "diagflat", "combinations", "_chunk_cat",
]

KEEP_CREATION = [
    "empty", "empty_like", "empty_strided", "empty_quantized", "empty_permuted",
    "zeros", "zeros_like", "ones", "ones_like",
    "full", "full_like",
    "arange", "linspace", "logspace",
    "eye", "scalar_tensor", "tensor",
    "new_empty", "new_empty_strided", "new_full", "new_zeros", "new_ones",
    "from_file", "_efficientzerotensor",
    "range",
]

KEEP_RANDOM = [
    "rand", "rand_like", "randn", "randn_like",
    "randint", "randint_like", "randperm",
    "normal", "uniform", "bernoulli",
    "multinomial", "poisson",
    "_standard_gamma", "_sample_dirichlet",
    "native_dropout",
    "random_", "exponential_", "geometric_", "cauchy_", "binomial",
    "_dirichlet_grad",
]

KEEP_AUTOGRAD = [
    "_backward", "set_data", "data", "is_leaf", "output_nr", "_version",
    "requires_grad_", "retain_grad", "retains_grad",
    "_fw_primal", "_make_dual", "_unpack_dual",
    "_new_zeros_with_same_feature_meta", "_has_same_storage_numel",
    "detach", "detach_", "detach_copy",
]

KEEP_TYPE_CAST = [
    "_cast_Byte", "_cast_Char", "_cast_Double", "_cast_Float",
    "_cast_Int", "_cast_Long", "_cast_Short", "_cast_Half",
    "to", "_to_copy", "_autocast_to_reduced_precision", "_autocast_to_full_precision",
]

KEEP_COPY_CLONE = [
    "clone", "copy_", "copy", "fill_", "fill", "zero_",
]

KEEP_INFRA = [
    "_assert_async", "_assert_scalar", "_assert_tensor_metadata",
    "_functional_assert_scalar", "_functional_assert_async",
    "_print", "sym_constrain_range", "sym_constrain_range_for_size",
    "_functional_sym_constrain_range", "_functional_sym_constrain_range_for_size",
    "_make_dep_token",
    "is_floating_point", "is_complex", "is_conj", "is_neg",
    "is_nonzero", "is_same_size", "is_signed", "is_inference",
    "is_contiguous", "is_pinned", "is_set_to", "is_distributed",
    "get_device",
    "size", "stride", "dim", "numel", "storage_offset",
    "is_coalesced",
    "pin_memory", "_pin_memory", "record_stream",
    "resize_", "resize_as_", "_resize_output_",
    "set_", "storage",
    "_unsafe_view", "_unsafe_index", "_unsafe_masked_index",
    "_unsafe_index_put", "_unsafe_masked_index_put_accumulate",
    "item", "_local_scalar_dense",
    "_debug_has_internal_overlap",
    "_test_check_tensor", "_test_autograd_multiple_dispatch",
    "_test_serialization_subcmul",
    "rename_", "rename", "align_to", "align_as", "align_tensors",
    "lift", "lift_fresh", "lift_fresh_copy",
    "_has_compatible_shallow_copy_type",
    "_validate_compressed_sparse_indices",
    "result_type", "can_cast", "promote_types",
    "_to_functional_tensor", "_from_functional_tensor",
    "_sync", "_functionalize_are_all_mutations_hidden_from_autograd",
    "_functionalize_has_data_mutation", "_functionalize_mark_mutation_hidden_from_autograd",
    "_functionalize_are_all_mutations_under_no_grad_or_inference_mode",
    "_functionalize_was_storage_changed", "_functionalize_has_metadata_mutation",
    "_functionalize_replace", "_functionalize_commit_update",
    "_functionalize_enable_reapply_views",
    "segment_reduce", "_segment_reduce_backward",
    # Misc utilities that infra may depend on
    "unbind", "unbind_copy", "meshgrid", "cartesian_prod",
    "complex", "polar", "argwhere", "block_diag",
    "nan_to_num", "nan_to_num_",
    "type_as", "ravel", "adjoint", "matrix_H", "numpy_T",
    "_index_put_impl_", "_lazy_clone", "_is_zerotensor",
    "_copy_from", "_copy_from_and_resize", "_to_cpu",
    "_reshape_copy", "_reshape_from_tensor", "_stack",
    "_shape_as_tensor", "_dim_arange",
    "_add_batch_dim", "_remove_batch_dim",
    "_propagate_xla_data",
    "sym_size", "sym_stride", "sym_numel", "sym_storage_offset", "sym_is_contiguous",
    "_test_ambiguous_defaults", "_test_functorch_fallback",
    "_test_optional_filled_intlist", "_test_optional_floatlist",
    "_test_optional_intlist", "_test_parallel_materialize",
    "_test_string_default", "_test_warn_in_autograd",
    "refine_names",
    "dense_dim",
    "values", "values_copy", "indices", "indices_copy",
    "row_indices", "row_indices_copy", "ccol_indices", "ccol_indices_copy",
    "is_vulkan_available", "_nnpack_available",
    # Histogram/bincount (general tensor ops)
    "histc", "histogram", "histogramdd", "bincount",
    "_histogramdd_bin_edges", "_histogramdd_from_bin_cts", "_histogramdd_from_bin_tensors",
    "isin",
    # Diff/gradient (reduction-like)
    "diff", "gradient", "trapezoid", "trapz", "cumulative_trapezoid",
    "vander",
    # Sobol/quasi-random
    "_sobol_engine_draw", "_sobol_engine_ff_",
    "_sobol_engine_initialize_state_", "_sobol_engine_scramble_",
    # Philox random
    "_philox_key_split", "_philox_normal_", "_philox_uniform_",
    # Sparse metadata queries (may be needed by infra)
    "_dimI", "_dimV", "_to_dense",
    # Test ops
    "_foobar",
]

# FHE-specific ops (ALWAYS keep)
KEEP_FHE = [
    "encrypt", "encode", "pre_encode",
    "add_mod", "sub_mod", "mul_mod",
    "add_scalar_mod", "sub_scalar_mod",
    "add_pt_broadcast", "add_pt_pairwise",
    "automorphism_transform", "batched_pairwise_mac", "cpmul_broadcast_pt",
    "extend_ciphertext", "mod_raise", "moddown", "modup",
    "drop_last_element_and_scale", "hash_tensor",
    "innerproduct", "innerproduct_broadcast_cipher",
]


def extract_op_base_name(func_line: str) -> str:
    """Extract base op name from a func line like '- func: add.Tensor(...) -> ...'"""
    m = re.match(r"- func:\s+(\w+)", func_line)
    if m:
        return m.group(1)
    return ""


def parse_ops(yaml_path: Path) -> list[str]:
    """Extract all op func lines from native_functions.yaml."""
    ops = []
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("- func:"):
                ops.append(line)
    return ops


def classify_op(func_line: str) -> tuple[str, str]:
    """Classify an op as (KEEP/DELETE, category)."""
    base_name = extract_op_base_name(func_line)
    if not base_name:
        return "KEEP", "unparseable"

    if base_name in KEEP_FHE:
        return "KEEP", "fhe"

    # Explicit KEEP overrides (false positives from substring matching)
    force_keep = {
        "detach": "autograd", "detach_": "autograd", "detach_copy": "autograd",
        "unfold": "shape", "unfold_backward": "shape", "unfold_copy": "shape",
        "slice_inverse": "shape",
    }
    if base_name in force_keep:
        return "KEEP", force_keep[base_name]

    # Check DELETE patterns FIRST (more specific, e.g. max_pool before max)
    delete_patterns = {
        "nn": ["conv", "pool", "batch_norm", "dropout", "relu", "gelu", "silu",
               "sigmoid", "tanh",
               "softmax", "log_softmax", "cross_entropy", "nll_loss", "layer_norm",
               "rnn", "lstm", "gru", "embedding", "upsample", "grid_sample",
               "pixel_shuffle", "pixel_unshuffle", "unfold", "fold", "linear",
               "bilinear", "cosine_similarity", "pdist", "cdist", "pad",
               "interpolat", "adaptive_avg", "adaptive_max", "avg_pool", "max_pool",
               "fractional_max", "lp_pool", "reflection_pad", "replication_pad",
               "elu", "leaky_relu", "prelu", "rrelu", "celu", "hardswish",
               "hardtanh", "hardsigmoid", "mish", "log_sigmoid", "threshold",
               "group_norm", "instance_norm", "native_batch_norm", "native_layer_norm",
               "native_group_norm", "_batch_norm", "cudnn_batch_norm", "miopen",
               "huber_loss", "smooth_l1_loss", "mse_loss", "l1_loss",
               "binary_cross_entropy", "margin_ranking_loss", "hinge_embedding_loss",
               "multi_margin_loss", "multilabel_margin_loss", "soft_margin_loss",
               "triplet_margin_loss", "ctc_loss", "poisson_nll_loss",
               "gaussian_nll_loss", "one_hot", "scaled_dot_product_attention",
               "_scaled_dot_product", "flash_attention", "_efficient_attention",
               "_transform_bias_rescale_qkv", "multi_head_attention",
               "channel_shuffle", "native_channel_shuffle",
               "feature_dropout", "feature_alpha_dropout", "alpha_dropout",
               "glu", "glu_backward", "glu_backward_jvp", "glu_jvp",
               "hardshrink", "hardshrink_backward",
               "softplus", "softplus_backward",
               "softshrink", "softshrink_backward",
               "selu", "selu_",
               "_fused_sdp_choice",
               "_masked_scale",
               ],
        "linalg": ["matmul", "bmm", "addmm", "mm", "mv", "dot", "vdot", "cross",
                   "linalg", "svd", "cholesky", "solve", "inverse", "det", "eig",
                   "lstsq", "lu_", "lu_solve", "lu_unpack", "pinverse", "matrix_rank",
                   "matrix_power", "qr", "triangular_solve", "ormqr", "geqrf", "orgqr",
                   "_linalg", "inner", "outer", "tensordot", "chain_matmul", "multi_dot",
                   "addr", "addbmm", "addmv", "ger", "baddbmm",
                   "matrix_exp", "linalg_", "_linalg_",
                   ],
        "special_math": ["special_", "bessel", "polygamma", "digamma", "lgamma",
                         "erf", "erfinv", "erfc", "erfcx", "i0", "i1",
                         "spherical_bessel", "airy", "chebyshev", "hermite",
                         "laguerre", "legendre", "shifted_chebyshev", "zeta",
                         "xlog1py", "xlogy", "entr", "multigammaln", "gammainc", "ndtr",
                         ],
        "sparse": ["sparse", "_sparse", "to_sparse", "crow_indices", "col_indices",
                   "ccol_indices", "row_indices", "coalesce", "_nnz", "_indices", "_values",
                   "indices", "values", "dense_dim", "sparse_dim",
                   "sparse_coo", "sparse_csr", "sparse_csc", "sparse_bsr", "sparse_bsc",
                   ],
        "foreach": ["_foreach_"],
        "fft": ["fft_", "stft", "istft", "bartlett", "blackman", "hamming", "hann", "kaiser"],
        "quantized": ["quantize", "dequantize", "fake_quantize", "q_scale", "q_zero_point",
                      "q_per_channel", "int_repr", "_make_per_tensor", "_make_per_channel",
                      "quantized"],
        "cudnn": ["_cudnn", "cudnn_"],
        "nested": ["nested", "_nested"],
        "mkldnn": ["mkldnn", "_mkldnn"],
        "optimizer": ["_fused_adam", "_fused_adamw", "_fused_sgd", "_fused_adagrad",
                      "_amp_foreach_non_finite_check", "_amp_update_scale"],
        "fbgemm": ["fbgemm_"],
        "misc_delete": ["_rowwise_prune", "_saturate_weight_to_fp16",
                        "_dyn_quant_pack", "choose_qparams_optimized",
                        "_fused_moving_avg_obs", "fused_moving_avg_obs",
                        "_cslt_compress", "_spdiags", "_spsolve",
                        "nuclear_norm", "frobenius_norm",
                        "pairwise_distance", "kl_div",
                        "_euclidean_dist",
                        "_trilinear",
                        "_weight_norm", "rms_norm", "_fused_rms_norm",
                        "_transformer_encoder_layer_fwd",
                        "col2im", "im2col",
                        "_cufft_clear_plan_cache", "_cufft_get_plan_cache",
                        "_cufft_set_plan_cache",
                        "chalf",
                        "affine_grid_generator",
                        "_choose_qparams_per_tensor", "qscheme",
                        ],
    }

    for category, patterns in delete_patterns.items():
        for pattern in patterns:
            if base_name.startswith(pattern) or f"_{pattern}" in base_name:
                return "DELETE", category

    # Special cases for linalg that don't start with the pattern
    linalg_exact = {"mm", "mv", "dot", "vdot", "bmm", "matmul", "addmm", "addbmm",
                    "addmv", "addr", "ger", "baddbmm", "chain_matmul", "multi_dot",
                    "tensordot", "inner", "outer", "matrix_exp", "einsum", "kron",
                    "logdet", "slogdet", "dist", "hspmm", "smm", "sspaddmm"}
    if base_name in linalg_exact:
        return "DELETE", "linalg"

    # Now check KEEP patterns (after DELETE, so specific DELETE patterns win)
    all_keep = {
        "reduction": KEEP_REDUCTION,
        "indexing": KEEP_INDEXING,
        "arithmetic": KEEP_ARITHMETIC,
        "comparison": KEEP_COMPARISON,
        "shape": KEEP_SHAPE,
        "creation": KEEP_CREATION,
        "random": KEEP_RANDOM,
        "autograd": KEEP_AUTOGRAD,
        "type_cast": KEEP_TYPE_CAST,
        "copy_clone": KEEP_COPY_CLONE,
        "infra": KEEP_INFRA,
        "fhe": KEEP_FHE,
    }

    for category, patterns in all_keep.items():
        for pattern in patterns:
            if base_name == pattern or base_name.startswith(pattern + "_") or base_name.startswith(pattern + "."):
                return "KEEP", category
            if base_name == pattern + "_":
                return "KEEP", category

    # Check for _out suffix variants of keep ops
    if base_name.endswith("_out"):
        stem = base_name[:-4]
        for category, patterns in all_keep.items():
            if stem in patterns:
                return "KEEP", category

    # If not matched by any pattern, default to KEEP (conservative)
    return "KEEP", "unclassified"


def main():
    ops = parse_ops(YAML_PATH)
    print(f"Total ops parsed: {len(ops)}")

    keep_ops = []
    delete_ops = []
    keep_by_cat = {}
    delete_by_cat = {}

    for func_line in ops:
        decision, category = classify_op(func_line)
        base_name = extract_op_base_name(func_line)
        if decision == "KEEP":
            keep_ops.append(base_name)
            keep_by_cat.setdefault(category, []).append(base_name)
        else:
            delete_ops.append(base_name)
            delete_by_cat.setdefault(category, []).append(base_name)

    # Output files
    out_dir = Path(__file__).parent
    with open(out_dir / "keep_ops.txt", "w") as f:
        for op in sorted(set(keep_ops)):
            f.write(op + "\n")
    with open(out_dir / "delete_ops.txt", "w") as f:
        for op in sorted(set(delete_ops)):
            f.write(op + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"KEEP: {len(keep_ops)} ops ({len(keep_ops)*100//len(ops)}%)")
    print(f"DELETE: {len(delete_ops)} ops ({len(delete_ops)*100//len(ops)}%)")
    print(f"{'='*60}")

    print(f"\n--- KEEP by category ---")
    for cat in sorted(keep_by_cat.keys()):
        print(f"  {cat:20s}: {len(keep_by_cat[cat]):4d}")

    print(f"\n--- DELETE by category ---")
    for cat in sorted(delete_by_cat.keys()):
        print(f"  {cat:20s}: {len(delete_by_cat[cat]):4d}")

    # Show unclassified KEEP ops for review
    if "unclassified" in keep_by_cat:
        print(f"\n--- Unclassified KEEP ops (review these) ---")
        for op in sorted(set(keep_by_cat["unclassified"])):
            print(f"  {op}")


if __name__ == "__main__":
    main()
