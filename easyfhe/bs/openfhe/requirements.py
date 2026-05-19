from __future__ import annotations

from .internal.approx_plan import bootstrap_approx_depth as _bootstrap_approx_depth_from_plan


def context_requirements(
    *,
    log_n,
    log_bs_slots,
    level_budget,
    secret_key_dist="SPARSE_TERNARY",
    dim1=None,
):
    """Return ``(extra_depth, rotation_keys, plaintexts)`` needed for OpenFHE bootstrap."""

    return (
        bootstrap_depth(log_bs_slots, level_budget, secret_key_dist),
        required_rotations(
            log_n,
            log_bs_slots,
            level_budget,
            secret_key_dist=secret_key_dist,
            dim1=dim1,
        ),
        required_plaintexts(log_n, log_bs_slots, level_budget, dim1=dim1),
    )


def bootstrap_depth(log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY"):
    """Return extra CKKS depth needed for these OpenFHE bootstraps."""

    params = _normalize_params(log_bs_slots, level_budget, None)
    if not params:
        return 0
    max_budget = max((budget for _, budget, _ in params), key=sum)
    return (
        bootstrap_approx_depth(secret_key_dist)
        + int(max_budget[0])
        + int(max_budget[1])
    )


def bootstrap_approx_depth(secret_key_dist="SPARSE_TERNARY"):
    return _bootstrap_approx_depth_from_plan(str(secret_key_dist))


def required_rotations(
    log_n,
    log_bs_slots,
    level_budget,
    *,
    secret_key_dist="SPARSE_TERNARY",
    dim1=None,
):
    """Return OpenFHE bootstrap rotation keys."""

    ring_dim = 1 << int(log_n)
    result = []
    for bs_slots, budget, dims in _normalize_params(log_bs_slots, level_budget, dim1):
        result.extend(_bootstrap_rotation_indices(ring_dim, bs_slots, budget, secret_key_dist, dims))
    return _unique_preserve_order(result)


def required_plaintexts(log_n, log_bs_slots, level_budget, *, dim1=None):
    """Return names of bootstrap plaintext constants that will be generated."""

    from .internal.rotations import linear_transform_plan

    ring_dim = 1 << int(log_n)
    result = []
    for bs_slots, budget, dims in _normalize_params(log_bs_slots, level_budget, dim1):
        slots = 1 << int(bs_slots)
        c2s_plan = linear_transform_plan("C2S", slots, budget[0], ring_dim, dims[0])
        s2c_plan = linear_transform_plan("S2C", slots, budget[1], ring_dim, dims[1])
        result.extend(_plaintext_names(c2s_plan))
        result.extend(_plaintext_names(s2c_plan))
    return _unique_preserve_order(result)


def _plaintext_names(plan):
    names = []
    log_slots = int(plan.slots).bit_length() - 1
    for loop_pos, level in enumerate(plan.loop_range):
        if loop_pos == len(plan.loop_range) - 1 and plan.rem:
            giant_step = plan.giant_step_rem
            baby_step = plan.baby_step_rem
            num_rotations = plan.num_rotations_rem
        else:
            giant_step = plan.giant_step
            baby_step = plan.baby_step
            num_rotations = plan.num_rotations
        for i in range(baby_step):
            start = giant_step * i
            for j in range(giant_step):
                index = start + j
                names.append(f"{plan.direction}_{log_slots}_{level}_{index}")
    return names


def _bootstrap_rotation_indices(ring_dim, log_bs_slots, level_budget, secret_key_dist, dim1):
    from .internal.rotations import bootstrap_rotation_indices

    return bootstrap_rotation_indices(
        ring_dim,
        log_bs_slots,
        level_budget,
        secret_key_dist,
        dim1,
    )


def _normalize_params(log_bs_slots, level_budget, dim1):
    slots_values = _as_sequence(log_bs_slots)
    budget_values = _normalize_budget_sequence(level_budget, len(slots_values))
    dim_values = _normalize_dim_sequence(dim1, len(slots_values))
    if len(slots_values) != len(budget_values):
        raise ValueError(
            "log_bs_slots and level_budget must describe the same number of bootstrap parameter sets"
        )
    if len(slots_values) != len(dim_values):
        raise ValueError(
            "log_bs_slots and dim1 must describe the same number of bootstrap parameter sets"
        )
    return tuple(
        (int(slots), _normalize_pair(budget, "level_budget"), _normalize_pair(dims, "dim1"))
        for slots, budget, dims in zip(slots_values, budget_values, dim_values)
    )


def _normalize_budget_sequence(value, count):
    if value is None:
        return ()
    if _is_pair(value):
        return tuple(value for _ in range(count))
    return tuple(value)


def _normalize_dim_sequence(value, count):
    if value is None:
        return tuple((0, 0) for _ in range(count))
    if _is_pair(value):
        return tuple(value for _ in range(count))
    return tuple(value)


def _as_sequence(value):
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        iterator = iter(value)
    except TypeError:
        return (value,)
    return tuple(iterator)


def _is_pair(value):
    if isinstance(value, (str, bytes)):
        return False
    try:
        return len(value) == 2 and all(_is_scalar(item) for item in value)
    except TypeError:
        return False


def _is_scalar(value):
    return isinstance(value, (int, float)) or not hasattr(value, "__iter__")


def _normalize_pair(value, name):
    if value is None or len(value) != 2:
        raise ValueError(f"bootstrap {name} must have two entries, got {value}")
    return (int(value[0]), int(value[1]))


def _unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
