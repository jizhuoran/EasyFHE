from __future__ import annotations


def generate_bootstrap_constants(
    crypto_context,
    log_bs_slots,
    level_budget,
    max_levels_remaining=None,
    *,
    dim1=None,
):
    from .internal.constants import generate_bootstrap_constants as _generate

    return _generate(
        crypto_context,
        int(log_bs_slots),
        _normalize_pair(level_budget, "level_budget"),
        maxLevelsRemaining=max_levels_remaining,
        dim1=_normalize_pair(dim1 or (0, 0), "dim1"),
    )


def _normalize_pair(value, name):
    if value is None or len(value) != 2:
        raise ValueError(f"bootstrap {name} must have two entries, got {value}")
    return (int(value[0]), int(value[1]))
