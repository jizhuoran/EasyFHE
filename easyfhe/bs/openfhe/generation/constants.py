from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from easyfhe.fhe.constants import ConstantBundle
from easyfhe.fhe.ops.encoding import PreparedPlaintext
from .plan import compile_flat_ps_plan, get_bootstrap_approx_plan
from .precompute import BootstrapPrecompute
from .rotations import (
    bootstrap_required_rotations,
    bootstrap_transform_schedule,
    normalize_bootstrap_strategy,
)
from .types import BootstrapPlan, BootstrapTransformPlan, BootstrapTransformStep


def _round_half_away_from_zero(number, ndigits=0):
    multiplier = 10**ndigits
    if number > 0:
        return math.floor(number * multiplier + 0.5) / multiplier
    if number < 0:
        return math.ceil(number * multiplier - 0.5) / multiplier
    return 0.0


@dataclass(frozen=True)
class _BootstrapConfig:
    log_bs_slots: int
    level_budget: list[int]
    dim1: list[int]
    strategy: str
    max_levels_remaining: int

    @property
    def slots(self):
        return 1 << self.log_bs_slots


def _normalize_level_budget(level_budget):
    if level_budget is None or len(level_budget) != 2:
        raise ValueError(f"bootstrap level_budget must have two entries, got {level_budget}")
    budget = [int(level_budget[0]), int(level_budget[1])]
    if budget[0] == 1 or budget[1] == 1:
        raise NotImplementedError(
            "OpenFHE bootstrap does not support the linear-transform route; "
            f"both level_budget entries must be greater than 1, got {tuple(budget)}"
        )
    return budget


def _normalize_dim1(dim1):
    if dim1 is None:
        return [0, 0]
    if len(dim1) != 2:
        raise ValueError(f"bootstrap dim1 must have two entries, got {dim1}")
    return [int(dim1[0]), int(dim1[1])]


def _normalize_strategy(strategy):
    return normalize_bootstrap_strategy(strategy)


def _resolve_max_levels_remaining(max_levels_remaining):
    if max_levels_remaining is not None:
        return int(max_levels_remaining)
    raise ValueError(
        "generate_bootstrap_constants requires max_levels_remaining. "
        "Use easyfhe.bs.openfhe.depth(...) to compute the extra bootstrap depth "
        "before creating CKKSContextSpec(depth=...)."
    )


def _make_generation_config(log_bs_slots, level_budget, max_levels_remaining, dim1, strategy):
    return _BootstrapConfig(
        log_bs_slots=int(log_bs_slots),
        level_budget=_normalize_level_budget(level_budget),
        dim1=_normalize_dim1(dim1),
        strategy=_normalize_strategy(strategy),
        max_levels_remaining=_resolve_max_levels_remaining(max_levels_remaining),
    )


def _make_precompute(crypto_context, log_bs_slots):
    return BootstrapPrecompute(
        crypto_context.N,
        int(log_bs_slots),
        0,
        crypto_context.secretKeyDist,
    )


def _unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _run_bootstrap_setup(crypto_context, plan):
    precompute = _make_precompute(crypto_context, plan.log_bs_slots)
    precompute.eval_bootstrap_setup(
        crypto_context,
        plan.level_budget,
        plan.dim1,
        plan.slots,
        0,
    )
    return precompute


def _compute_bootstrap_scalars(crypto_context, plan, k):
    q0 = float(crypto_context.moduliQ_scalar[0])
    p = crypto_context.dcrtBits
    deg = int(_round_half_away_from_zero(math.log2(q0) - p))

    N = crypto_context.N
    correction_factor = 9

    correction = correction_factor - deg
    correction_scale = 2.0**(-correction)
    post = 2.0**deg
    pre = 1.0 / post
    post_scalar = round(post)
    constant_eval_mult = pre * (1.0 / (k * N))
    cor_factor = 1 << round(correction)

    return dict(
        deg=deg,
        correction_factor=correction_factor,
        correction=correction,
        correction_scale=correction_scale,
        pre=pre,
        post_scalar=post_scalar,
        constant_eval_mult=constant_eval_mult,
        cor_factor=cor_factor,
    )


def _constants_from_precompute(plan, precompute, required_rotations, crypto_context):
    scalars = _compute_bootstrap_scalars(crypto_context, plan, precompute.k)
    approx_plan = get_bootstrap_approx_plan(crypto_context.secretKeyDist)
    approx_eval_plan = compile_flat_ps_plan(approx_plan.ps_root)
    (
        approx_tail_scalar_names,
        approx_constant_scalar_names,
        approx_q_highest_scalar_names,
    ) = _register_approx_scalars(scalars, approx_eval_plan)
    chebyshev_neg_one_scalar_name = _register_chebyshev_scalars(scalars)
    double_angle_scalar_names = _register_double_angle_scalars(scalars, crypto_context.secretKeyDist)
    c2s_plan = bootstrap_transform_schedule("C2S", plan.slots, plan.level_budget[0], precompute.N, plan.dim1[0])
    s2c_plan = bootstrap_transform_schedule("S2C", plan.slots, plan.level_budget[1], precompute.N, plan.dim1[1])
    plaintext_names = {
        "C2S": _name_table(c2s_plan),
        "S2C": _name_table(s2c_plan),
    }
    vectors = {
        **_flatten_table(plaintext_names["C2S"], precompute.m_U0hatTPreFFT, plan.slots),
        **_flatten_table(plaintext_names["S2C"], precompute.m_U0PreFFT, plan.slots),
    }
    c2s_runtime_plan = _build_runtime_transform_plan(c2s_plan, plaintext_names["C2S"], vectors)
    s2c_runtime_plan = _build_runtime_transform_plan(s2c_plan, plaintext_names["S2C"], vectors)
    bootstrap_plan = BootstrapPlan(
        log_bs_slots=plan.log_bs_slots,
        level_budget=tuple(plan.level_budget),
        dim1=tuple(plan.dim1),
        strategy=plan.strategy,
        max_levels_remaining=plan.max_levels_remaining,
        c2s_plan=c2s_runtime_plan,
        s2c_plan=s2c_runtime_plan,
        approx_eval_plan=approx_eval_plan,
        approx_tail_scalar_names=approx_tail_scalar_names,
        approx_constant_scalar_names=approx_constant_scalar_names,
        approx_q_highest_scalar_names=approx_q_highest_scalar_names,
        chebyshev_neg_one_scalar_name=chebyshev_neg_one_scalar_name,
        double_angle_iterations=approx_plan.double_angle_iterations,
        double_angle_scalar_names=double_angle_scalar_names,
        required_rotations=tuple(required_rotations),
    )
    constants = ConstantBundle(scalars=scalars, vectors=vectors)
    return constants, bootstrap_plan


def _register_approx_scalars(scalars, flat):
    zero_name = "approx.zero"
    scalars[zero_name] = 0.0

    tail_names = []
    constant_names = {}
    q_highest_names = {}

    for spec in flat.small_specs:
        const_name = "approx." + ".".join((*spec.scalar_path, "const"))
        scalars[const_name] = float(spec.const_value)
        constant_names[spec.scalar_path] = const_name

        if spec.q_highest_scalar_value is not None:
            name = "approx." + ".".join((*spec.scalar_path, "highest"))
            scalars[name] = int(spec.q_highest_scalar_value)
            q_highest_names[spec.scalar_path] = name

    for spec in flat.tail_specs:
        row = []
        for index in range(1, flat.tail_max_deg + 1):
            if index <= spec.deg:
                name = "approx." + ".".join((*spec.scalar_path, str(index)))
                scalars[name] = float(spec.coefficients[index])
                row.append(name)
            else:
                row.append(zero_name)
        tail_names.append(tuple(row))

    return tuple(tail_names), constant_names, q_highest_names


def _register_chebyshev_scalars(scalars):
    name = "approx.chebyshev.neg_one"
    scalars[name] = -1.0
    return name


def _register_double_angle_scalars(scalars, secret_key_dist):
    approx_plan = get_bootstrap_approx_plan(secret_key_dist)
    names = []
    for j in range(1, approx_plan.double_angle_iterations + 1):
        name = f"approx.double_angle.{j}"
        scalars[name] = -1.0 / math.pow(
            2.0 * math.pi,
            math.pow(2.0, j - approx_plan.double_angle_iterations),
        )
        names.append(name)
    return tuple(names)


def _flatten_table(name_table, table, default_slots):
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


def _build_runtime_transform_plan(plan, name_table, vectors):
    steps = []
    log_slots = int(plan.slots).bit_length() - 1
    for loop_pos, level in enumerate(plan.loop_range):
        if loop_pos == len(plan.loop_range) - 1 and plan.rem:
            giant_step = int(plan.giant_step_rem)
            baby_step = int(plan.baby_step_rem)
        else:
            giant_step = int(plan.giant_step)
            baby_step = int(plan.baby_step)

        names = tuple(
            name_table[int(level)][group * giant_step + index]
            for group in range(baby_step)
            for index in range(giant_step)
        )
        slots = {
            int(np.asarray(vectors[name]).reshape(-1).size)
            for name in names
        }
        if len(slots) != 1:
            raise ValueError(f"{plan.direction} constants have mixed slots at level {level}: {sorted(slots)}")
        batch_name = f"{plan.direction}_{log_slots}_{int(level)}_batch"
        vectors[batch_name] = np.stack([vectors.pop(name) for name in names], axis=0)
        giant_offsets = tuple(int(offset) for offset in plan.rot_out[level][:baby_step])
        giant_offset = int(giant_offsets[1]) - int(giant_offsets[0]) if len(giant_offsets) > 1 else 0
        steps.append(
            BootstrapTransformStep(
                level=int(level),
                input_offsets=tuple(int(offset) for offset in plan.rot_in[level][:giant_step]),
                plaintext_name=batch_name,
                plaintext_slots=slots.pop(),
                giant_offset=giant_offset,
                baby_step=baby_step,
                giant_step=giant_step,
            )
        )
    return BootstrapTransformPlan(direction=plan.direction, steps=tuple(steps))


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
    max_levels_remaining=None,
    *,
    dim1=None,
    baby_step=None,
    strategy="double_hoist",
):
    if dim1 is not None and baby_step is not None:
        raise ValueError("bootstrap generate accepts either dim1 or baby_step, not both")
    dim1 = baby_step if baby_step is not None else dim1

    return _build_bootstrap_constants(
        crypto_context,
        int(log_bs_slots),
        _normalize_public_pair(level_budget, "level_budget"),
        max_levels_remaining=max_levels_remaining,
        dim1=_normalize_public_pair_or_scalar(dim1 or (0, 0), "dim1"),
        strategy=strategy,
    )


def _normalize_public_pair(value, name):
    if value is None or len(value) != 2:
        raise ValueError(f"bootstrap {name} must have two entries, got {value}")
    return (int(value[0]), int(value[1]))


def _normalize_public_pair_or_scalar(value, name):
    if isinstance(value, (int, float)):
        return (int(value), int(value))
    return _normalize_public_pair(value, name)


def _build_bootstrap_constants(
    crypto_context,
    log_bs_slots,
    level_budget,
    max_levels_remaining=None,
    dim1=None,
    strategy="double_hoist",
):
    plan = _make_generation_config(log_bs_slots, level_budget, max_levels_remaining, dim1, strategy)
    precompute = _run_bootstrap_setup(crypto_context, plan)
    required_rotations = bootstrap_required_rotations(
        crypto_context.N,
        plan.log_bs_slots,
        plan.level_budget,
        plan.dim1,
        plan.strategy,
    )
    return _constants_from_precompute(plan, precompute, required_rotations, crypto_context)
