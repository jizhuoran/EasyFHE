from __future__ import annotations

from .plan import bootstrap_approx_depth as _bootstrap_approx_depth_from_plan
from .constants import _normalize_bootstrap_mode
from .rotations import bootstrap_required_rotations, normalize_bootstrap_strategy


def bootstrap_depth(log_bs_slots, level_budget, secret_key_dist="SPARSE_TERNARY", bootstrap_mode="modraise_first"):
    """Return extra CKKS depth needed for these OpenFHE bootstraps."""

    params = _normalize_params(log_bs_slots, level_budget, None)
    if not params:
        return 0
    _reject_linear_transform_budget(params)
    mode = _normalize_bootstrap_mode(bootstrap_mode)
    max_budget = max((budget for _, budget, _ in params), key=lambda budget: _budget_depth(budget, mode))
    return (
        bootstrap_approx_depth(secret_key_dist)
        + _budget_depth(max_budget, mode)
    )


def bootstrap_approx_depth(secret_key_dist="SPARSE_TERNARY"):
    return _bootstrap_approx_depth_from_plan(str(secret_key_dist))


def _budget_depth(budget, bootstrap_mode):
    depth = int(budget[0]) + int(budget[1])
    if bootstrap_mode == "stc_first":
        depth += int(budget[1])
    return depth


def required_rotations(
    log_n,
    log_bs_slots,
    level_budget,
    *,
    strategy="double_hoist",
    dim1=None,
):
    """Return OpenFHE bootstrap rotation keys."""

    ring_dim = 1 << int(log_n)
    result = []
    params = _normalize_params(log_bs_slots, level_budget, dim1)
    _reject_linear_transform_budget(params)
    strategy = normalize_bootstrap_strategy(strategy)
    for bs_slots, budget, dims in params:
        result.extend(bootstrap_required_rotations(ring_dim, bs_slots, budget, dims, strategy))
    return _unique_preserve_order(result)


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
