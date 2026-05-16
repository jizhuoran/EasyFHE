"""Public CKKS-oriented FHE frontend for EasyFHE."""

from ._public_api import PUBLIC_API as _PUBLIC_API
from . import config, utils
from .ciphertext import PreparedPlaintext
from .constants import ConstantBundle
from .context import Context
from .ops import (
    CipherState,
    align_to,
    double_hoist_rotate_sum,
    encode,
    fast_rotate,
    fast_rotate_ext,
    fused_broadcast_mac,
    fused_pairwise_mac,
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
    moddown_from_ext,
    make_plaintext,
    prepare_plaintext,
    slot_resize,
)
from .runtime.instrumentation import NullInstrumentation, OpInstrumentation, profile
from .runtime.options import RuntimeOptions
from .runtime.spec import (
    CKKSContextSpec,
    generate_context,
)


__all__ = list(_PUBLIC_API)


def __dir__():
    return sorted(__all__)
