from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapFFTParams:
    level_budget: int
    layers_coll: int
    layers_rem: int
    num_rotations: int
    baby_step: int
    giant_step: int
    num_rotations_rem: int
    baby_step_rem: int
    giant_step_rem: int


@dataclass(frozen=True)
class BootstrapTransformSchedule:
    direction: str
    slots: int
    level_budget: int
    rem: int
    num_rotations: int
    baby_step: int
    giant_step: int
    num_rotations_rem: int
    baby_step_rem: int
    giant_step_rem: int
    loop_range: tuple[int, ...]
    rot_in: tuple[tuple[int, ...], ...]
    rot_out: tuple[tuple[int, ...], ...]


def reduce_rotation(index, slots):
    slots = int(slots)
    index = int(index)

    if slots & (slots - 1) == 0:
        n = int(math.log2(slots))
        if index >= 0:
            return index - ((index >> n) << n)
        return index + slots + ((abs(index) >> n) << n)

    return (slots + index % slots) % slots


def select_layers(log_slots, budget):
    log_slots = int(log_slots)
    budget = int(budget)
    layers = int(math.ceil(log_slots / budget))
    rows = int(log_slots // layers)
    rem = log_slots % layers

    dim = rows + 1 if rem != 0 else rows
    if dim < budget:
        layers -= 1
        rows = log_slots // layers
        rem = log_slots - rows * layers
        dim = rows + 1 if rem != 0 else rows

        while dim != budget:
            rows -= 1
            rem = log_slots - rows * layers
            dim = rows + 1 if rem != 0 else rows

    return [int(layers), int(rows), int(rem)]


def _default_giant_step(num_rotations, layers):
    if num_rotations > 7:
        return 1 << (int(layers / 2) + 2)
    return 1 << (int(layers / 2) + 1)


def _split_from_baby_step(num_rotations, baby_step):
    total = int(num_rotations) + 1
    baby_step = min(int(baby_step), total)
    if baby_step <= 0:
        raise ValueError(f"bootstrap baby_step must be positive, got {baby_step}")
    if total % baby_step != 0:
        raise ValueError(
            f"bootstrap baby_step={baby_step} must divide transform width {total}"
        )
    return baby_step, total // baby_step


def collapsed_fft_params(slots, level_budget, dim1=0, baby_step=None):
    requested_baby_step = baby_step
    dims = select_layers(int(math.log2(slots)), level_budget)
    layers_collapse = dims[0]
    rem_collapse = dims[2]
    flag_rem = 1 if rem_collapse != 0 else 0

    num_rotations = (1 << (layers_collapse + 1)) - 1
    num_rotations_rem = (1 << (rem_collapse + 1)) - 1

    if requested_baby_step is not None and int(requested_baby_step) > 0:
        baby_step, giant_step = _split_from_baby_step(num_rotations, requested_baby_step)
    elif dim1 == 0 or dim1 > num_rotations:
        giant_step = _default_giant_step(num_rotations, layers_collapse)
        baby_step = (num_rotations + 1) // giant_step
    else:
        giant_step = dim1
        baby_step = (num_rotations + 1) // giant_step
    baby_step_rem = 0
    giant_step_rem = 0

    if flag_rem:
        if requested_baby_step is not None and int(requested_baby_step) > 0:
            baby_step_rem, giant_step_rem = _split_from_baby_step(num_rotations_rem, requested_baby_step)
        elif dim1 != 0 and dim1 <= num_rotations_rem:
            giant_step_rem = dim1
        else:
            giant_step_rem = _default_giant_step(num_rotations_rem, rem_collapse)
        if baby_step_rem == 0:
            baby_step_rem = (num_rotations_rem + 1) // giant_step_rem

    return BootstrapFFTParams(
        int(level_budget),
        int(layers_collapse),
        int(rem_collapse),
        int(num_rotations),
        int(baby_step),
        int(giant_step),
        int(num_rotations_rem),
        int(baby_step_rem),
        int(giant_step_rem),
    )


def normalize_bootstrap_strategy(strategy):
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


def bootstrap_transform_schedule(direction, slots, level_budget, ring_dim, dim1=0, baby_step=None):
    direction = str(direction).upper()
    if direction not in ("C2S", "S2C"):
        raise ValueError(f"unknown bootstrap transform direction: {direction}")

    slots = int(slots)
    level_budget = int(level_budget)
    params = collapsed_fft_params(slots, level_budget, int(dim1), baby_step=baby_step)
    flag_rem = int(params.layers_rem != 0)
    cycl_order = int(ring_dim) << 1

    if direction == "C2S":
        exp_fn = lambda s: (s - flag_rem) * params.layers_coll + params.layers_rem
        loop_range = tuple(range(level_budget - 1, -1, -1))
        populated_range = range(level_budget - 1, flag_rem - 1, -1)
        in_div = slots
        rem_index = 0
    else:
        exp_fn = lambda s: s * params.layers_coll
        loop_range = tuple(range(level_budget))
        populated_range = range(level_budget - flag_rem)
        in_div = cycl_order // 4
        rem_index = level_budget - flag_rem

    center = (params.num_rotations + 1) // 2
    rem_center = (params.num_rotations_rem + 1) // 2

    rot_in = []
    rot_out = []
    for s in range(level_budget):
        use_rem = flag_rem and ((direction == "C2S" and s == 0) or (direction == "S2C" and s == level_budget - 1))
        in_size = params.num_rotations_rem + 1 if use_rem else params.num_rotations + 1
        rot_in.append([0] * in_size)
        rot_out.append([0] * (params.baby_step + params.baby_step_rem))

    for s in populated_range:
        exp = exp_fn(s)
        for j in range(params.giant_step):
            rot_in[s][j] = reduce_rotation((j - center + 1) << exp, in_div)
        for i in range(params.baby_step):
            rot_out[s][i] = reduce_rotation((params.giant_step * i) << exp, cycl_order // 4)

    if flag_rem:
        exp = 0 if direction == "C2S" else exp_fn(rem_index)
        for j in range(params.giant_step_rem):
            rot_in[rem_index][j] = reduce_rotation((j - rem_center + 1) << exp, in_div)
        for i in range(params.baby_step_rem):
            rot_out[rem_index][i] = reduce_rotation((params.giant_step_rem * i) << exp, cycl_order // 4)

    return BootstrapTransformSchedule(
        direction=direction,
        slots=slots,
        level_budget=level_budget,
        rem=params.layers_rem,
        num_rotations=params.num_rotations,
        baby_step=params.baby_step,
        giant_step=params.giant_step,
        num_rotations_rem=params.num_rotations_rem,
        baby_step_rem=params.baby_step_rem,
        giant_step_rem=params.giant_step_rem,
        loop_range=loop_range,
        rot_in=tuple(tuple(int(value) for value in row) for row in rot_in),
        rot_out=tuple(tuple(int(value) for value in row) for row in rot_out),
    )


def bootstrap_required_rotations(ring_dim, log_bs_slots, level_budget, dim1=None, strategy="double_hoist", baby_step=None):
    """Return bootstrap rotations plus the conjugation key rotation."""
    dim1 = [0, 0] if dim1 is None else dim1
    if baby_step is None:
        baby_step = [None, None]
    elif isinstance(baby_step, (int, float)):
        baby_step = [baby_step, baby_step]
    ring_dim = int(ring_dim)
    slots = 1 << int(log_bs_slots)
    c2s_schedule = bootstrap_transform_schedule("C2S", slots, level_budget[0], ring_dim, dim1[0], baby_step=baby_step[0])
    s2c_schedule = bootstrap_transform_schedule("S2C", slots, level_budget[1], ring_dim, dim1[1], baby_step=baby_step[1])
    rotations = []
    for schedule in (c2s_schedule, s2c_schedule):
        rotations.extend(_schedule_required_rotations(schedule, strategy, ring_dim))
    rotations.extend(_sparse_bootstrap_rotations(ring_dim, slots))
    rotations.append((ring_dim << 1) - 1)
    return _unique_preserve_order(rotations)


def _schedule_required_rotations(schedule, strategy, ring_dim):
    result = []
    strategy = normalize_bootstrap_strategy(strategy)
    for loop_pos, level in enumerate(schedule.loop_range):
        if loop_pos == len(schedule.loop_range) - 1 and schedule.rem:
            giant_step = schedule.giant_step_rem
            baby_step = schedule.baby_step_rem
        else:
            giant_step = schedule.giant_step
            baby_step = schedule.baby_step

        result.extend(schedule.rot_in[level][:giant_step])
        rot_out = schedule.rot_out[level][:baby_step]
        if strategy == "double_hoist":
            result.extend(rot_out)
        else:
            result.extend(_single_giant_rotation_key(rot_out, ring_dim))
    return [int(rotation) for rotation in result if int(rotation) != 0]


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


def _unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
