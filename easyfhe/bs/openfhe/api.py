from __future__ import annotations

from .generation.constants import generate_bootstrap_constants
from .generation.plan import describe_flat_ps_plan
from .generation.requirements import bootstrap_depth, required_rotations
from .generation.types import BootstrapPlan


def depth(*, log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY", bootstrap_mode="modraise_first"):
    return bootstrap_depth(log_bs_slots, level_budget, secret_key_dist, bootstrap_mode)


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
    max_levels_remaining=None,
    dim1=None,
    baby_step=None,
    strategy="double_hoist",
    bootstrap_mode="modraise_first",
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
        bootstrap_mode=bootstrap_mode,
    )
    return constants, plan


def describe_plan(plan):
    return describe_flat_ps_plan(plan.approx_eval_plan, plan)


def bootstrap(cipher, crypto_context, constants, plan, *, L0):
    from .runtime import homo_bootstrap

    return homo_bootstrap(cipher, crypto_context, constants, plan, L0=L0)
