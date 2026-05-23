from __future__ import annotations

from .generation.constants import BootstrapPlan, generate_bootstrap_constants
from .generation.requirements import bootstrap_depth, required_rotations


def depth(*, log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY"):
    return bootstrap_depth(log_bs_slots, level_budget, secret_key_dist)


def plan_rot_keys(
    *,
    log_n,
    log_bs_slots,
    level_budget,
    secret_key_dist="SPARSE_TERNARY",
    dim1=None,
):
    return tuple(
        required_rotations(
            log_n,
            log_bs_slots,
            level_budget,
            secret_key_dist=secret_key_dist,
            dim1=dim1,
        )
    )


def generate(
    crypto_context,
    *,
    log_bs_slots,
    level_budget,
    max_levels_remaining=None,
    dim1=None,
    baby_step=None,
    strategy="double_hoist",
):
    if max_levels_remaining is None:
        raise ValueError("generate requires max_levels_remaining when crypto_context is provided")
    constants, plan = generate_bootstrap_constants(
        crypto_context,
        log_bs_slots,
        level_budget,
        max_levels_remaining=max_levels_remaining,
        dim1=dim1,
        baby_step=baby_step,
        strategy=strategy,
    )
    return constants, plan


def bootstrap(cipher, crypto_context, constants, plan, *, L0):
    from .runtime import homo_bootstrap

    return homo_bootstrap(cipher, crypto_context, constants, plan, L0=L0)
