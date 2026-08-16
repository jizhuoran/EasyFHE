"""Public CKKS-oriented FHE frontend for EasyFHE."""

from ._public_api import PUBLIC_API as _PUBLIC_API
from .ciphertext import Cipher, CipherState, EncodedScalar
from .client import Client
from .constants import ConstantBundle, PackedRaw, encode_scalar
from .context import Context, ContextParams
from .ops import (
    align_to,
    fast_rotate,
    giant_rotate_sum,
    hoisted_mac_sum,
    hoisted_mac_sum_rescale,
    grouped_pairwise_mac,
    grouped_pairwise_mac_rescale,
    grouped_scalar_weighted_acc,
    homo_add,
    homo_add_inplace,
    homo_add_pt,
    homo_add_pt_inplace,
    homo_add_scalar,
    homo_add_scalar_inplace,
    homo_mul_no_relin,
    homo_mul_i,
    homo_relinearize,
    homo_mul_relin,
    homo_mul_relin_rescale_postop,
    homo_mul_relin_rescale_add_pt,
    homo_mul_relin_rescale_add_scalar,
    homo_mul_pt,
    homo_mul_pt_inplace,
    homo_mul_pt_rescale,
    homo_mul_scalar,
    homo_mul_scalar_inplace,
    homo_mul_scalar_rescale,
    moddown_from_ext,
    prepare_hoisted_baby_rotations,
    homo_rotate,
    homo_rotate_add,
    normalize_scale,
    rescale,
    homo_sub,
    homo_sub_inplace,
    homo_sub_scalar,
    homo_sub_scalar_inplace,
    expand_slots,
    fold_slots,
    pack_cipher_batch,
    unpack_cipher_batch,
    sum_cipher_batch,
)
from .context_factory import CKKSContextSpec, PrimeChainPlan, generate_client_context, plan_prime_chain


__all__ = list(_PUBLIC_API)


def __dir__():
    return sorted(__all__)
