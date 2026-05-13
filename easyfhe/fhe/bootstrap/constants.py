from __future__ import annotations

import math
from dataclasses import dataclass

from .rotations import bootstrap_auto_index_map, bootstrap_rotation_indices, linear_transform_plan
from .precompute_context import BsContext


def _round_half_away_from_zero(number, ndigits=0):
    multiplier = 10**ndigits
    if number > 0:
        return math.floor(number * multiplier + 0.5) / multiplier
    if number < 0:
        return math.ceil(number * multiplier - 0.5) / multiplier
    return 0.0


@dataclass(frozen=True)
class BootstrapPlan:
    log_bs_slots: int
    level_budget: list[int]
    dim1: list[int]
    max_levels_remaining: int

    @property
    def slots(self):
        return 1 << self.log_bs_slots


@dataclass(frozen=True)
class BootstrapConstants:
    log_bs_slots: int
    level_budget: list[int]
    dim1: list[int]
    max_levels_remaining: int
    c2s: list[list[object]]
    s2c: list[list[object]]
    c2s_plan: object
    s2c_plan: object
    required_rotations: list[int]
    # precomputed scalars
    deg: int
    correction_factor: int
    correction: int
    pre: float
    scalar: int
    constant_eval_mult: float
    cor_factor: int

    def get(self, direction: str, level: int, index: int):
        direction = direction.upper()
        if direction == "C2S":
            table = self.c2s
        elif direction == "S2C":
            table = self.s2c
        else:
            raise ValueError(f"Unknown bootstrap constant direction: {direction}")
        return table[int(level)][int(index)]


def _normalize_level_budget(level_budget):
    if level_budget is None or len(level_budget) != 2:
        raise ValueError(f"bootstrap level_budget must have two entries, got {level_budget}")
    return [int(level_budget[0]), int(level_budget[1])]


def _normalize_dim1(dim1):
    if dim1 is None:
        return [0, 0]
    if len(dim1) != 2:
        raise ValueError(f"bootstrap dim1 must have two entries, got {dim1}")
    return [int(dim1[0]), int(dim1[1])]


def _resolve_max_levels_remaining(crypto_context, maxLevelsRemaining):
    if maxLevelsRemaining is not None:
        return int(maxLevelsRemaining)
    raise ValueError(
        "generate_bootstrap_constants requires maxLevelsRemaining. "
        "Use fhe.bootstrap_depth(maxLevelsRemaining, bootstrap_specs, secret_key_dist) "
        "when creating CKKSContextSpec(depth=...)."
    )


def _make_plan(crypto_context, log_bs_slots, level_budget, maxLevelsRemaining, dim1):
    return BootstrapPlan(
        log_bs_slots=int(log_bs_slots),
        level_budget=_normalize_level_budget(level_budget),
        dim1=_normalize_dim1(dim1),
        max_levels_remaining=_resolve_max_levels_remaining(crypto_context, maxLevelsRemaining),
    )


def _make_bs_context(crypto_context, log_bs_slots):
    return BsContext(
        crypto_context.N,
        int(log_bs_slots),
        0,
        crypto_context.secretKeyDist,
    )


def _bootstrap_auto_index_map(crypto_context, plan):
    auto_idx_to_rot_idx = bootstrap_auto_index_map(
        crypto_context.N,
        plan.log_bs_slots,
        plan.level_budget,
        crypto_context.secretKeyDist,
        plan.dim1,
    )
    return {int(auto_idx): int(rot_idx) for auto_idx, rot_idx in auto_idx_to_rot_idx.items()}


def _unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _bootstrap_rotation_indices(crypto_context, plan):
    return _unique_preserve_order(
        bootstrap_rotation_indices(
            crypto_context.N,
            plan.log_bs_slots,
            plan.level_budget,
            crypto_context.secretKeyDist,
            plan.dim1,
        )
    )


def _run_bootstrap_setup(crypto_context, plan):
    bs_context = _make_bs_context(crypto_context, plan.log_bs_slots)
    bs_context.eval_bootstrap_setup(
        crypto_context,
        plan.level_budget,
        plan.dim1,
        plan.slots,
        0,
        plan.max_levels_remaining,
    )
    return bs_context


def _compute_bootstrap_scalars(crypto_context, plan, k):
    q0 = float(crypto_context.moduliQ_scalar[0])
    p = crypto_context.dcrtBits
    deg = int(_round_half_away_from_zero(math.log2(q0) - p))

    rescale_tech = crypto_context.rescaleTech
    M = crypto_context.M
    N = crypto_context.N
    slots = plan.slots

    if rescale_tech == "FLEXIBLEAUTO":
        tmp = _round_half_away_from_zero(-0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1)
        if tmp < 7:
            correction_factor = 7
        elif tmp > 13:
            correction_factor = 13
        else:
            correction_factor = int(tmp)
    else:
        correction_factor = 9

    correction = correction_factor - deg
    post = 2.0**deg
    pre = 1.0 / post
    scalar = round(post)
    constant_eval_mult = pre * (1.0 / (k * N))
    cor_factor = 1 << round(correction)

    return dict(
        deg=deg,
        correction_factor=correction_factor,
        correction=correction,
        pre=pre,
        scalar=scalar,
        constant_eval_mult=constant_eval_mult,
        cor_factor=cor_factor,
    )


def _constants_from_bs_context(plan, bs_context, required_rotations, crypto_context):
    scalars = _compute_bootstrap_scalars(crypto_context, plan, bs_context.k)
    return BootstrapConstants(
        log_bs_slots=plan.log_bs_slots,
        level_budget=plan.level_budget,
        dim1=plan.dim1,
        max_levels_remaining=plan.max_levels_remaining,
        c2s=bs_context.m_U0hatTPreFFT,
        s2c=bs_context.m_U0PreFFT,
        c2s_plan=linear_transform_plan("C2S", plan.slots, plan.level_budget[0], bs_context.N, plan.dim1[0]),
        s2c_plan=linear_transform_plan("S2C", plan.slots, plan.level_budget[1], bs_context.N, plan.dim1[1]),
        required_rotations=required_rotations,
        deg=scalars["deg"],
        correction_factor=scalars["correction_factor"],
        correction=scalars["correction"],
        pre=scalars["pre"],
        scalar=scalars["scalar"],
        constant_eval_mult=scalars["constant_eval_mult"],
        cor_factor=scalars["cor_factor"],
    )


def generate_bootstrap_constants(
    crypto_context,
    log_bs_slots,
    level_budget,
    maxLevelsRemaining=None,
    dim1=None,
    ensure_rotation_keys=True,
):
    plan = _make_plan(crypto_context, log_bs_slots, level_budget, maxLevelsRemaining, dim1)
    required_rotations = _bootstrap_rotation_indices(crypto_context, plan)
    if ensure_rotation_keys:
        crypto_context.ensure_rotation_keys([required_rotations])

    bs_context = _run_bootstrap_setup(crypto_context, plan)
    constants = _constants_from_bs_context(plan, bs_context, required_rotations, crypto_context)

    return constants
