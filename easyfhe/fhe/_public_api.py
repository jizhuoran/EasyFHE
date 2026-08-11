"""Stable public API allowlist for :mod:`easyfhe.fhe`.

Names listed here are the symbols EasyFHE documents as supported for external
users via ``import easyfhe.fhe as fhe``. Submodules may still be importable as
Python implementation details, but they are outside the stable API contract
unless they are re-exported here.
"""

CONTEXT_API = (
    "Cipher",
    "CipherState",
    "EncodedScalar",
    "CKKSContextSpec",
    "Client",
    "Context",
    "ContextParams",
    "PrimeChainPlan",
    "generate_client_context",
    "plan_prime_chain",
)

CONSTANT_API = (
    "ConstantBundle",
    "PackedRaw",
    "encode_scalar",
)

ENCODING_API = ()

ALIGNMENT_API = (
    "align_to",
    "normalize_scale",
    "rescale",
)

BASIC_OPS_API = (
    "homo_add",
    "homo_add_inplace",
    "homo_add_pt",
    "homo_add_pt_inplace",
    "homo_add_scalar",
    "homo_add_scalar_inplace",
    "homo_mul_no_relin",
    "homo_mul_i",
    "homo_relinearize",
    "homo_mul_relin",
    "homo_mul_relin_rescale_postop",
    "homo_mul_relin_rescale_add_pt",
    "homo_mul_relin_rescale_add_scalar",
    "homo_mul_pt",
    "homo_mul_pt_inplace",
    "homo_mul_pt_rescale",
    "homo_mul_scalar",
    "homo_mul_scalar_inplace",
    "homo_mul_scalar_rescale",
    "homo_rotate",
    "homo_rotate_add",
    "homo_sub",
    "homo_sub_inplace",
    "homo_sub_scalar",
    "homo_sub_scalar_inplace",
)

HOISTED_OPS_API = (
    "fast_rotate",
    "grouped_pairwise_mac",
    "grouped_pairwise_mac_rescale",
    "grouped_scalar_weighted_acc",
    "giant_rotate_sum",
    "hoisted_mac_sum",
    "hoisted_mac_sum_rescale",
    "moddown_from_ext",
    "sum_cipher_batch",
)

SHAPE_API = (
    "expand_slots",
    "fold_slots",
    "pack_cipher_batch",
    "unpack_cipher_batch",
)

PUBLIC_API = (
    *CONTEXT_API,
    *CONSTANT_API,
    *ENCODING_API,
    *ALIGNMENT_API,
    *BASIC_OPS_API,
    *HOISTED_OPS_API,
    *SHAPE_API,
)
