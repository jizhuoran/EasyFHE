from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from easyfhe.fhe.ciphertext import PreparedPlaintext
from easyfhe.fhe.constants import ConstantBundle
from .approx_plan import degree, get_bootstrap_approx_plan
from .rotations import linear_transform_plan
from .precompute_context import BsContext


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


@dataclass(frozen=True)
class BootstrapTransformStep:
    level: int
    input_offsets: tuple[int, ...]
    plaintext_name: str
    plaintext_slots: int
    giant_offset: int
    baby_step: int
    giant_step: int


@dataclass(frozen=True)
class BootstrapTransformPlan:
    direction: str
    steps: tuple[BootstrapTransformStep, ...]


@dataclass(frozen=True)
class BootstrapPlan:
    log_bs_slots: int
    level_budget: tuple[int, int]
    dim1: tuple[int, int]
    strategy: str
    max_levels_remaining: int
    c2s_plan: BootstrapTransformPlan
    s2c_plan: BootstrapTransformPlan
    approx_scalar_names: dict[tuple[str, ...], tuple[str, ...]]
    double_angle_scalar_names: tuple[str, ...]
    required_rotations: tuple[int, ...]

    @property
    def slots(self):
        return 1 << self.log_bs_slots


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


def _normalize_strategy(strategy):
    strategy = str(strategy or "double_hoist").lower()
    aliases = {
        "double_hoist": "double_hoist",
        "double-hoist": "double_hoist",
        "ext_double_hoist": "double_hoist",
        "normal_giant": "normal_giant",
        "normal-giant": "normal_giant",
        "ext_normal_giant": "normal_giant",
        "normal_bsgs": "normal_bsgs",
        "normal-bsgs": "normal_bsgs",
    }
    try:
        return aliases[strategy]
    except KeyError as exc:
        raise ValueError(
            "bootstrap strategy must be one of: double_hoist, normal_giant, normal_bsgs"
        ) from exc


def _resolve_max_levels_remaining(crypto_context, maxLevelsRemaining):
    if maxLevelsRemaining is not None:
        return int(maxLevelsRemaining)
    raise ValueError(
        "generate_bootstrap_constants requires maxLevelsRemaining. "
        "Use easyfhe.bs.openfhe.generate(...) without a context to get extra depth, "
        "then add it to maxLevelsRemaining "
        "when creating CKKSContextSpec(depth=...)."
    )


def _make_plan(crypto_context, log_bs_slots, level_budget, maxLevelsRemaining, dim1, strategy):
    return _BootstrapConfig(
        log_bs_slots=int(log_bs_slots),
        level_budget=_normalize_level_budget(level_budget),
        dim1=_normalize_dim1(dim1),
        strategy=_normalize_strategy(strategy),
        max_levels_remaining=_resolve_max_levels_remaining(crypto_context, maxLevelsRemaining),
    )


def _make_bs_context(crypto_context, log_bs_slots):
    return BsContext(
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


def _bootstrap_rotation_indices(crypto_context, plan, c2s_plan, s2c_plan):
    rotations = []
    for transform_plan in (c2s_plan, s2c_plan):
        rotations.extend(_transform_required_rotations(transform_plan, plan.strategy, crypto_context.N))
    rotations.extend(_sparse_bootstrap_rotations(crypto_context.N, plan.slots))
    rotations.append((int(crypto_context.N) << 1) - 1)
    return _unique_preserve_order(rotations)


def _transform_required_rotations(transform_plan, strategy, ring_dim):
    rotations = []
    for loop_pos, level in enumerate(transform_plan.loop_range):
        if loop_pos == len(transform_plan.loop_range) - 1 and transform_plan.rem:
            giant_step = transform_plan.giant_step_rem
            baby_step = transform_plan.baby_step_rem
        else:
            giant_step = transform_plan.giant_step
            baby_step = transform_plan.baby_step

        rotations.extend(transform_plan.rot_in[level][:giant_step])
        rot_out = transform_plan.rot_out[level][:baby_step]
        if strategy == "double_hoist":
            rotations.extend(rot_out)
        else:
            rotations.extend(_single_giant_rotation_key(rot_out, ring_dim))
    return [int(rotation) for rotation in rotations if int(rotation) != 0]


def _single_giant_rotation_key(offsets, ring_dim):
    offsets = tuple(int(offset) for offset in offsets)
    if len(offsets) <= 1:
        return ()
    modulus = int(ring_dim) // 2
    step = (offsets[1] - offsets[0]) % modulus
    return () if step == 0 else (step,)


def _sparse_bootstrap_rotations(ring_dim, slots):
    ring_dim = int(ring_dim)
    slots = int(slots)
    count = int(math.log2(ring_dim // (2 * slots)))
    return [(1 << step) * slots for step in range(count)]


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


def _constants_from_bs_context(plan, bs_context, required_rotations, crypto_context):
    scalars = _compute_bootstrap_scalars(crypto_context, plan, bs_context.k)
    approx_scalar_names = _register_approx_scalars(scalars, crypto_context.secretKeyDist)
    double_angle_scalar_names = _register_double_angle_scalars(scalars, crypto_context.secretKeyDist)
    c2s_plan = linear_transform_plan("C2S", plan.slots, plan.level_budget[0], bs_context.N, plan.dim1[0])
    s2c_plan = linear_transform_plan("S2C", plan.slots, plan.level_budget[1], bs_context.N, plan.dim1[1])
    plaintext_names = {
        "C2S": _name_table(c2s_plan),
        "S2C": _name_table(s2c_plan),
    }
    vectors = {
        **_flatten_table(plaintext_names["C2S"], bs_context.m_U0hatTPreFFT, bs_context.N, plan.slots),
        **_flatten_table(plaintext_names["S2C"], bs_context.m_U0PreFFT, bs_context.N, plan.slots),
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
        approx_scalar_names=approx_scalar_names,
        double_angle_scalar_names=double_angle_scalar_names,
        required_rotations=tuple(required_rotations),
    )
    constants = ConstantBundle(scalars=scalars, vectors=vectors)
    return constants, bootstrap_plan


def _register_approx_scalars(scalars, secret_key_dist):
    names = {}
    approx_plan = get_bootstrap_approx_plan(secret_key_dist)

    def add(path, coefficients, size=None):
        path = tuple(path)
        deg = _coeff_degree(coefficients, size)
        item_names = []
        for index, value in enumerate(coefficients[1 : deg + 1], start=1):
            name = "approx." + ".".join((*path, str(index)))
            scalars[name] = float(value)
            item_names.append(name)
        names[path] = tuple(item_names)

    def walk(node, path):
        add((*path, "c"), node.divcs_q)
        add((*path, "q"), node.divqr_q, node.k)
        add((*path, "s"), node.s2, node.k)
        if node.q_node is not None:
            walk(node.q_node, (*path, "q_node"))
        if node.s_node is not None:
            walk(node.s_node, (*path, "s_node"))

    walk(approx_plan.ps_root, ("root",))
    return names


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


def _coeff_degree(coefficients, size=None):
    if size is None:
        return degree(coefficients)
    truncated = np.copy(coefficients[: int(size)])
    truncated.resize(int(size), refcheck=False)
    return degree(truncated)


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
    maxLevelsRemaining=None,
    dim1=None,
    strategy="double_hoist",
):
    plan = _make_plan(crypto_context, log_bs_slots, level_budget, maxLevelsRemaining, dim1, strategy)
    bs_context = _run_bootstrap_setup(crypto_context, plan)
    c2s_plan = linear_transform_plan("C2S", plan.slots, plan.level_budget[0], bs_context.N, plan.dim1[0])
    s2c_plan = linear_transform_plan("S2C", plan.slots, plan.level_budget[1], bs_context.N, plan.dim1[1])
    required_rotations = _bootstrap_rotation_indices(crypto_context, plan, c2s_plan, s2c_plan)
    return _constants_from_bs_context(plan, bs_context, required_rotations, crypto_context)
