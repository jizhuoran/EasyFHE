from __future__ import annotations


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
    from .internal.constants import generate_bootstrap_constants as _generate

    if dim1 is not None and baby_step is not None:
        raise ValueError("bootstrap generate accepts either dim1 or baby_step, not both")
    dim1 = baby_step if baby_step is not None else dim1

    return _generate(
        crypto_context,
        int(log_bs_slots),
        _normalize_pair(level_budget, "level_budget"),
        maxLevelsRemaining=max_levels_remaining,
        dim1=_normalize_pair_or_scalar(dim1 or (0, 0), "dim1"),
        strategy=strategy,
    )


def _normalize_pair(value, name):
    if value is None or len(value) != 2:
        raise ValueError(f"bootstrap {name} must have two entries, got {value}")
    return (int(value[0]), int(value[1]))


def _normalize_pair_or_scalar(value, name):
    if isinstance(value, (int, float)):
        return (int(value), int(value))
    return _normalize_pair(value, name)
