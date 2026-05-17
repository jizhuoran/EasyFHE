from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from easyfhe.fhe.ciphertext import PreparedPlaintext
from easyfhe.fhe.constants import ConstantBundle
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


BootstrapConstants = ConstantBundle


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
        "Use easyfhe.bs.openfhe.generate(...) without a context to get extra depth, "
        "then add it to maxLevelsRemaining "
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
    c2s_plan = linear_transform_plan("C2S", plan.slots, plan.level_budget[0], bs_context.N, plan.dim1[0])
    s2c_plan = linear_transform_plan("S2C", plan.slots, plan.level_budget[1], bs_context.N, plan.dim1[1])
    plaintext_names = {
        "C2S": _name_table(c2s_plan),
        "S2C": _name_table(s2c_plan),
    }
    return ConstantBundle(
        info={
            "kind": "openfhe.bootstrap",
            "log_bs_slots": plan.log_bs_slots,
            "level_budget": plan.level_budget,
            "dim1": plan.dim1,
            "max_levels_remaining": plan.max_levels_remaining,
            "c2s_plan": c2s_plan,
            "s2c_plan": s2c_plan,
            "scalar_names": {
                "degree": "deg",
                "correction_factor": "correction_factor",
                "correction": "correction",
                "pre": "pre",
                "post_scalar": "scalar",
                "constant_eval_mult": "constant_eval_mult",
                "cor_factor": "cor_factor",
            },
            "plaintext_names": plaintext_names,
            "required_rotations": required_rotations,
        },
        scalars=scalars,
        vectors={
            **_flatten_table(plaintext_names["C2S"], bs_context.m_U0hatTPreFFT, bs_context.N, plan.slots),
            **_flatten_table(plaintext_names["S2C"], bs_context.m_U0PreFFT, bs_context.N, plan.slots),
        },
    )


def _flatten_table(name_table, table, ring_dim, default_slots):
    vectors = {}
    zero_cache = {}
    for level, row in enumerate(table):
        row_slots = _row_slots(row, default_slots)
        for index, name in enumerate(name_table[level]):
            value = row[index] if index < len(row) else None
            if value is None:
                value = _zero_vector(row_slots, zero_cache)
            vectors[name] = _raw_vector(value)
    return vectors


def _row_slots(row, default_slots):
    for value in row:
        if value is not None:
            return _value_slots(value, default_slots)
    return int(default_slots)


def _value_slots(value, default_slots):
    if isinstance(value, PreparedPlaintext):
        return int(value.slots)
    try:
        return int(np.asarray(value).reshape(-1).size)
    except Exception:
        return int(default_slots)


def _zero_vector(slots, zero_cache):
    slots = int(slots)
    key = slots
    if key not in zero_cache:
        zero_cache[key] = np.zeros(slots, dtype=np.complex128)
    return zero_cache[key]


def _raw_vector(value):
    if isinstance(value, PreparedPlaintext):
        value = value.values
    return np.asarray(value, dtype=np.complex128).reshape(-1)


def _name_table(plan):
    names = [[] for _ in range(int(plan.level_budget))]
    log_slots = int(plan.slots).bit_length() - 1
    for level, row_size in _plan_row_sizes(plan):
        names[int(level)] = [
            f"{plan.direction}_{log_slots}_{int(level)}_{int(index)}"
            for index in range(row_size)
        ]
    return names


def _plan_row_sizes(plan):
    rows = []
    for loop_pos, level in enumerate(plan.loop_range):
        if loop_pos == len(plan.loop_range) - 1 and plan.rem:
            row_size = int(plan.num_rotations_rem) + 1
        else:
            row_size = int(plan.num_rotations) + 1
        rows.append((level, row_size))
    return rows


def generate_bootstrap_constants(
    crypto_context,
    log_bs_slots,
    level_budget,
    maxLevelsRemaining=None,
    dim1=None,
):
    plan = _make_plan(crypto_context, log_bs_slots, level_budget, maxLevelsRemaining, dim1)
    required_rotations = _bootstrap_rotation_indices(crypto_context, plan)

    bs_context = _run_bootstrap_setup(crypto_context, plan)
    constants = _constants_from_bs_context(plan, bs_context, required_rotations, crypto_context)

    return constants
