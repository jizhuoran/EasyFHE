"""Public CKKS-oriented FHE frontend for EasyFHE."""

from ._public_api import PUBLIC_API as _PUBLIC_API
from .ciphertext import Cipher, CipherState
from .client import Client
from .constants import ConstantBundle, PackedRaw, UnpackedRaw
from .context import Context
from .ops.encoding import encode_stage1_packed
from .ops import (
    HOIST_EXT_DOUBLE_HOIST,
    HOIST_EXT_NORMAL,
    HOIST_NORMAL,
    align_to,
    fast_rotate,
    giant_rotate_sum,
    hoisted_mac_sum,
    grouped_pairwise_mac,
    grouped_scalar_weighted_acc,
    homo_add,
    homo_add_inplace,
    homo_add_pt,
    homo_add_pt_inplace,
    homo_add_scalar_double,
    homo_add_scalar_double_inplace,
    homo_add_scalar_int,
    homo_add_scalar_int_inplace,
    homo_mul_relin,
    homo_mul_relin_rescale_postop,
    homo_mul_relin_rescale_add_pt,
    homo_mul_relin_rescale_add_scalar,
    homo_mul_pt,
    homo_mul_pt_inplace,
    homo_mul_scalar_double,
    homo_mul_scalar_double_inplace,
    homo_mul_scalar_int,
    homo_mul_scalar_int_inplace,
    moddown_from_ext,
    homo_rotate,
    homo_rotate_add,
    reduce_noise_to_one,
    rescale_one_level,
    homo_sub,
    homo_sub_inplace,
    homo_sub_scalar_int,
    homo_sub_scalar_int_inplace,
    expand_slots,
    fold_slots,
    pack_cipher_batch,
    unpack_cipher_batch,
)
from .context_factory import CKKSContextSpec, generate_client_context


__all__ = list(_PUBLIC_API)


def __dir__():
    return sorted(__all__)
