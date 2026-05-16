from __future__ import annotations

from .constants import generate_bootstrap_constants
from .requirements import bootstrap_depth


def depth(*, log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY"):
    return bootstrap_depth(log_bs_slots, level_budget, secret_key_dist)


def generate(
    crypto_context,
    *,
    log_bs_slots,
    level_budget,
    max_levels_remaining=None,
    dim1=None,
):
    if max_levels_remaining is None:
        raise ValueError("generate requires max_levels_remaining when crypto_context is provided")
    constants = generate_bootstrap_constants(
        crypto_context,
        log_bs_slots,
        level_budget,
        max_levels_remaining=max_levels_remaining,
        dim1=dim1,
    )
    return _resolve_rotations(constants), constants


def _resolve_rotations(constants_or_rotations):
    info = getattr(constants_or_rotations, "info", None)
    if info is not None and "required_rotations" in info:
        return tuple(info["required_rotations"])
    return tuple(int(rotation) for rotation in (constants_or_rotations or ()))


def bootstrap(cipher, crypto_context, constants, *, L0):
    from .internal.runtime import homo_bootstrap

    return homo_bootstrap(cipher, crypto_context, constants, L0=L0)
