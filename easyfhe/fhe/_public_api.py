"""Stable public API allowlist for :mod:`easyfhe.fhe`.

Names listed here are the symbols EasyFHE documents as supported for external
users via ``import easyfhe.fhe as fhe``. Submodules may still be importable as
Python implementation details, but they are outside the stable API contract
unless they are re-exported here.
"""

CONFIG_API = (
    "CKKSContextSpec",
    "RuntimeOptions",
)

RUNTIME_API = (
    "ConstantBundle",
    "Context",
    "PreparedPlaintext",
)

CONSTRUCTION_API = (
    "generate_context",
)

ENCODING_API = (
    "encode",
    "make_plaintext",
    "prepare_plaintext",
)

ALIGNMENT_API = (
    "CipherState",
    "align_to",
)

HOMOMORPHIC_OPS_API = (
    "double_hoist_rotate_sum",
    "fast_rotate",
    "fast_rotate_ext",
    "fused_broadcast_mac",
    "fused_pairwise_mac",
    "homo_add",
    "homo_add_pt",
    "homo_add_scalar_double",
    "homo_add_scalar_int",
    "homo_mul",
    "homo_mul_pt",
    "homo_mul_scalar_double",
    "homo_mul_scalar_int",
    "homo_rotate",
    "homo_square",
    "homo_sub",
    "moddown_from_ext",
    "slot_resize",
)

INSTRUMENTATION_API = (
    "NullInstrumentation",
    "OpInstrumentation",
    "profile",
)

MODULE_API = (
    "config",
    "utils",
)

PUBLIC_API = (
    *CONFIG_API,
    *RUNTIME_API,
    *CONSTRUCTION_API,
    *ENCODING_API,
    *ALIGNMENT_API,
    *HOMOMORPHIC_OPS_API,
    *INSTRUMENTATION_API,
    *MODULE_API,
)
