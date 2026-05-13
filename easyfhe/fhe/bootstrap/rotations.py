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
class LinearTransformPlan:
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


def collapsed_fft_params(slots, level_budget, dim1):
    dims = select_layers(int(math.log2(slots)), level_budget)
    layers_collapse = dims[0]
    rem_collapse = dims[2]
    flag_rem = 1 if rem_collapse != 0 else 0

    num_rotations = (1 << (layers_collapse + 1)) - 1
    num_rotations_rem = (1 << (rem_collapse + 1)) - 1

    if dim1 == 0 or dim1 > num_rotations:
        if num_rotations > 7:
            giant_step = 1 << (int(layers_collapse / 2) + 2)
        else:
            giant_step = 1 << (int(layers_collapse / 2) + 1)
    else:
        giant_step = dim1

    baby_step = (num_rotations + 1) // giant_step
    baby_step_rem = 0
    giant_step_rem = 0

    if flag_rem:
        if num_rotations_rem > 7:
            giant_step_rem = 1 << (int(rem_collapse / 2) + 2)
        else:
            giant_step_rem = 1 << (int(rem_collapse / 2) + 1)
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


def _add_bsgs_rotations(index_list, num_rotations, g, b, scaling_factor, mod_j, mod_i):
    half_rots = 1 - ((num_rotations + 1) // 2)
    for j in range(half_rots, g + half_rots):
        index_list.append(reduce_rotation(j * scaling_factor, mod_j))
    for i in range(b):
        index_list.append(reduce_rotation((g * i) * scaling_factor, mod_i))


def _find_fft_rotation_indices(level_budget_vec, dim1_vec, slots, cycl_order, budget_idx, dim1_idx, coeffs_to_slots):
    slots = int(slots)
    cycl_order = int(cycl_order)
    params = collapsed_fft_params(slots, level_budget_vec[budget_idx], dim1_vec[dim1_idx])

    flag_rem = 0 if params.layers_rem == 0 else 1
    if not coeffs_to_slots and params.level_budget < flag_rem:
        raise ValueError("levelBudget can not be less than flagRem")

    index_list = []
    index_list_size = (
        params.baby_step
        + params.giant_step
        - 2
        + params.baby_step_rem
        + params.giant_step_rem
        - 2
        + 1
        + cycl_order
    )
    if index_list_size < 0:
        raise ValueError("indexListSz can not be negative")

    if coeffs_to_slots:
        for s in range(params.level_budget - 1, flag_rem - 1, -1):
            scaling_factor = 1 << ((s - flag_rem) * params.layers_coll + params.layers_rem)
            _add_bsgs_rotations(
                index_list,
                params.num_rotations,
                params.giant_step,
                params.baby_step,
                scaling_factor,
                slots,
                cycl_order // 4,
            )
    else:
        for s in range(0, params.level_budget - flag_rem):
            scaling_factor = 1 << (s * params.layers_coll)
            _add_bsgs_rotations(
                index_list,
                params.num_rotations,
                params.giant_step,
                params.baby_step,
                scaling_factor,
                cycl_order // 4,
                cycl_order // 4,
            )

    if flag_rem:
        if coeffs_to_slots:
            _add_bsgs_rotations(
                index_list,
                params.num_rotations_rem,
                params.giant_step_rem,
                params.baby_step_rem,
                1,
                slots,
                cycl_order // 4,
            )
        else:
            s = params.level_budget - flag_rem
            scaling_factor = 1 << (s * params.layers_coll)
            _add_bsgs_rotations(
                index_list,
                params.num_rotations_rem,
                params.giant_step_rem,
                params.baby_step_rem,
                scaling_factor,
                cycl_order // 4,
                cycl_order // 4,
            )

    m = slots * 4
    if m != cycl_order:
        ratio = cycl_order // m
        j = 1
        while j < ratio:
            index_list.append(j * slots)
            j <<= 1

    return index_list


def coeffs_to_slots_rotation_indices(level_budget, dim1, slots, cycl_order):
    return _find_fft_rotation_indices(level_budget, dim1, slots, cycl_order, 0, 0, True)


def slots_to_coeffs_rotation_indices(level_budget, dim1, slots, cycl_order):
    return _find_fft_rotation_indices(level_budget, dim1, slots, cycl_order, 1, 1, False)


def linear_transform_plan(direction, slots, level_budget, ring_dim, dim1=0):
    direction = str(direction).upper()
    if direction not in ("C2S", "S2C"):
        raise ValueError(f"unknown bootstrap linear transform direction: {direction}")

    slots = int(slots)
    level_budget = int(level_budget)
    params = collapsed_fft_params(slots, level_budget, int(dim1))
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

    return LinearTransformPlan(
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


def bootstrap_core_rotation_indices(level_budget, dim1, slots, cycl_order):
    rotations = []
    rotations.extend(coeffs_to_slots_rotation_indices(level_budget, dim1, slots, cycl_order))
    rotations.extend(slots_to_coeffs_rotation_indices(level_budget, dim1, slots, cycl_order))
    rotations = set(rotations)
    rotations.discard(0)
    rotations.discard(cycl_order // 4)
    return list(sorted(rotations))


def _mod_inverse(value, modulus):
    old_r, r = int(value), int(modulus)
    old_s, s = 1, 0
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    if old_r != 1:
        raise ValueError(f"{value} is not invertible modulo {modulus}")
    return old_s % modulus


def find_auto_index_2n_complex(rot_index, cycl_order):
    rot_index = int(rot_index)
    cycl_order = int(cycl_order)
    if rot_index == 0:
        return 1
    if rot_index == cycl_order - 1:
        return cycl_order - 1

    generator = _mod_inverse(5, cycl_order) if rot_index < 0 else 5
    result = generator
    for _ in range(1, abs(rot_index)):
        result = (result * generator) % cycl_order
    return int(result)


def bootstrap_auto_index_map(ring_dim, log_bs_slots, level_budget, secret_key_dist, dim1=None):
    """Return the bootstrap automorphism-index to rotation-index map."""

    log_bs_slots = int(log_bs_slots)
    dim1 = [0, 0] if dim1 is None else dim1
    slots = 1 << log_bs_slots
    ring_dim = int(ring_dim)
    cycl_order = ring_dim << 1
    rotations = bootstrap_core_rotation_indices(level_budget, dim1, slots, cycl_order)
    half_ring_dim = ring_dim // 2

    auto_idx_to_rot_idx = {}
    for rot in rotations:
        adj_rot = int(rot)
        if adj_rot < 0:
            adj_rot = half_ring_dim - abs(adj_rot)
        auto_idx = int(find_auto_index_2n_complex(adj_rot, cycl_order))
        auto_idx_to_rot_idx[auto_idx] = adj_rot

    return auto_idx_to_rot_idx


def bootstrap_rotation_indices(ring_dim, log_bs_slots, level_budget, secret_key_dist, dim1=None):
    """Return bootstrap rotations plus the conjugation key rotation."""

    auto_idx_to_rot_idx = bootstrap_auto_index_map(
        ring_dim,
        log_bs_slots,
        level_budget,
        secret_key_dist,
        dim1,
    )
    rotations = list(auto_idx_to_rot_idx.values())
    rotations.append((int(ring_dim) << 1) - 1)
    return rotations
