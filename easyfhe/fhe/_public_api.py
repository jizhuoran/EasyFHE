"""Stable public API allowlist for :mod:`easyfhe.fhe`.

Names listed here are the symbols EasyFHE documents as supported for external
users via ``import easyfhe.fhe as fhe``. Submodules may still be importable as
Python implementation details, but they are outside the stable API contract
unless they are re-exported here.
"""

CONTEXT_API = (
    "Cipher",
    "CipherState",
    "CKKSContextSpec",
    "Client",
    "Context",
    "generate_client_context",
)

CONSTANT_API = (
    "ConstantBundle",
)

ALIGNMENT_API = (
    "align_to",
    "reduce_noise_to_one",
)

BASIC_OPS_API = (
    "homo_add",
    "homo_add_inplace",
    "homo_add_pt",
    "homo_add_pt_inplace",
    "homo_add_scalar_double",
    "homo_add_scalar_double_inplace",
    "homo_add_scalar_int",
    "homo_add_scalar_int_inplace",
    "homo_mul_relin",
    "homo_mul_relin_rescale_postop",
    "homo_mul_relin_rescale_add_pt",
    "homo_mul_relin_rescale_add_scalar",
    "homo_mul_pt",
    "homo_mul_pt_inplace",
    "homo_mul_scalar_double",
    "homo_mul_scalar_double_inplace",
    "homo_mul_scalar_int",
    "homo_mul_scalar_int_inplace",
    "homo_rotate",
    "homo_sub",
    "homo_sub_inplace",
    "homo_sub_scalar_int",
    "homo_sub_scalar_int_inplace",
)

HOISTED_OPS_API = (
    "fast_rotate",
    "grouped_pairwise_mac",
    "grouped_scalar_weighted_acc",
    "giant_rotate_sum",
    "hoisted_mac_sum",
    "moddown_from_ext",
)

SHAPE_API = (
    "slot_resize",
)

PUBLIC_API = (
    *CONTEXT_API,
    *CONSTANT_API,
    *ALIGNMENT_API,
    *BASIC_OPS_API,
    *HOISTED_OPS_API,
    *SHAPE_API,
)
