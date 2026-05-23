from __future__ import annotations

from .plan import bootstrap_approx_depth as _bootstrap_approx_depth_from_plan


def bootstrap_depth(log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY"):
    """Return extra CKKS depth needed for these OpenFHE bootstraps."""

    params = _normalize_params(log_bs_slots, level_budget, None)
    if not params:
        return 0
    _reject_linear_transform_budget(params)
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
    params = _normalize_params(log_bs_slots, level_budget, dim1)
    _reject_linear_transform_budget(params)
    for bs_slots, budget, dims in params:
        result.extend(_bootstrap_rotation_indices(ring_dim, bs_slots, budget, secret_key_dist, dims))
    return _unique_preserve_order(result)


def _bootstrap_rotation_indices(ring_dim, log_bs_slots, level_budget, secret_key_dist, dim1):
    from .rotations import bootstrap_rotation_indices

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


def _reject_linear_transform_budget(params):
    for _, budget, _ in params:
        if int(budget[0]) == 1 or int(budget[1]) == 1:
            raise NotImplementedError(
                "OpenFHE bootstrap does not support the linear-transform route; "
                f"both level_budget entries must be greater than 1, got {budget}"
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
