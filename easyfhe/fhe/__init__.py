"""Public CKKS-oriented FHE frontend for EasyFHE."""

from ._public_api import PUBLIC_API as _PUBLIC_API
from .constants import ConstantBundle
from .context import Context
from .ops import (
    CipherState,
    align_to,
    fast_rotate,
    giant_rotate_sum,
    hoisted_mac_sum,
    fused_broadcast_mac,
    fused_grouped_pairwise_mac,
    homo_add,
    homo_add_pt,
    homo_add_scalar_double,
    homo_add_scalar_int,
    homo_mul,
    homo_mul_pt,
    homo_mul_scalar_double,
    homo_mul_scalar_int,
    homo_rotate,
    homo_square,
    homo_sub,
    slot_resize,
)
from .runtime.options import RuntimeOptions
from .runtime.spec import (
    CKKSContextSpec,
    generate_context,
)


__all__ = list(_PUBLIC_API)


def __dir__():
    return sorted(__all__)
