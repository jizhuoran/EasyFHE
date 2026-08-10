"""Stable semantic operation exports.

Backend wrappers and implementation helpers stay in their submodules and are
not part of the supported ``easyfhe.fhe.ops`` surface.
"""

from .._public_api import ALIGNMENT_API, BASIC_OPS_API, ENCODING_API, HOISTED_OPS_API, SHAPE_API
from .alignment import align_to, reduce_noise_to_one, rescale_one_level
from .arithmetic import (
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
    homo_mul_no_relin,
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
    homo_sub,
    homo_sub_inplace,
    homo_sub_scalar_int,
    homo_sub_scalar_int_inplace,
)
from .rotation import (
    HOIST_EXT_DOUBLE_HOIST,
    HOIST_EXT_NORMAL,
    HOIST_NORMAL,
    fast_rotate,
    giant_rotate_sum,
    hoisted_mac_sum,
    homo_rotate_add,
    homo_rotate,
    moddown_from_ext,
)
from .layout import expand_slots, fold_slots, pack_cipher_batch, unpack_cipher_batch
from .encoding import encode_stage1_packed

__all__ = list((*ENCODING_API, *ALIGNMENT_API, *BASIC_OPS_API, *HOISTED_OPS_API, *SHAPE_API))
