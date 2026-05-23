from __future__ import annotations

from .generation.constants import generate_bootstrap_constants
from .generation.plan import describe_flat_ps_plan
from .generation.requirements import bootstrap_depth, required_rotations
from .generation.types import BootstrapPlan


def depth(*, log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY"):
    return bootstrap_depth(log_bs_slots, level_budget, secret_key_dist)


def plan_rot_keys(
    *,
    log_n,
    log_bs_slots,
    level_budget,
    strategy="double_hoist",
    dim1=None,
):
    return tuple(
        required_rotations(
            log_n,
            log_bs_slots,
            level_budget,
            strategy=strategy,
            dim1=dim1,
        )
    )


def generate(
    crypto_context,
    *,
    log_bs_slots,
    level_budget,
    post_bootstrap_levels=None,
    max_levels_remaining=None,
    dim1=None,
    baby_step=None,
    strategy="double_hoist",
):
    if post_bootstrap_levels is None:
        post_bootstrap_levels = max_levels_remaining
    if post_bootstrap_levels is None:
        raise ValueError("generate requires post_bootstrap_levels when crypto_context is provided")
    constants, plan = generate_bootstrap_constants(
        crypto_context,
        log_bs_slots,
        level_budget,
        post_bootstrap_levels=post_bootstrap_levels,
        dim1=dim1,
        baby_step=baby_step,
        strategy=strategy,
    )
    return constants, plan


def describe_plan(plan):
    return describe_flat_ps_plan(plan.approx_eval_plan, plan)


def bootstrap(cipher, crypto_context, constants, plan, *, L0, bootstrap_mode="modraise_first"):
    from .runtime import homo_bootstrap

    return homo_bootstrap(cipher, crypto_context, constants, plan, L0=L0, bootstrap_mode=bootstrap_mode)
