"""Stable public API allowlist for :mod:`easyfhe.fhe`.

Names listed here are the symbols EasyFHE documents as supported for external
users via ``import easyfhe.fhe as fhe``. Submodules may still be importable as
Python implementation details, but they are outside the stable API contract
unless they are re-exported here.
"""

CONTEXT_API = (
    "CKKSContextSpec",
    "Context",
    "RuntimeOptions",
    "generate_context",
)

CONSTANT_API = (
    "ConstantBundle",
)

ALIGNMENT_API = (
    "CipherState",
    "align_to",
)

BASIC_OPS_API = (
    "homo_add",
    "homo_add_pt",
    "homo_add_scalar_double",
    "homo_add_scalar_int",
    "homo_mul",
    "homo_mul_rescale_addpt",
    "homo_mul_rescale_addscalar",
    "homo_mul_pt",
    "homo_mul_scalar_double",
    "homo_mul_scalar_int",
    "homo_rotate",
    "homo_square",
    "homo_sub",
)

HOISTED_OPS_API = (
    "fast_rotate",
    "fused_broadcast_mac",
    "fused_grouped_pairwise_mac",
    "giant_rotate_sum",
    "hoisted_mac_sum",
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
