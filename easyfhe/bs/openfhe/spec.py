from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from easyfhe.fhe.ciphertext import CipherState

from .generation.requirements import bootstrap_depth, required_rotations


_BOOTSTRAP_MODES = frozenset(("modraise_first", "stc_first"))
_BOOTSTRAP_STRATEGIES = frozenset(("double_hoist", "normal_giant", "normal_bsgs"))


@dataclass(frozen=True)
class BootstrapSpec:
    """Reusable public configuration for one u64 CKKS bootstrap."""

    log_slots: int
    level_budget: tuple[int, int]
    output_levels: int
    strategy: str = "double_hoist"
    mode: str = "modraise_first"
    dim1: tuple[int, int] | None = None
    baby_step: tuple[int, int] | None = None
    raise_to_limbs: int | None = None

    def __post_init__(self):
        log_slots = int(self.log_slots)
        if log_slots <= 0:
            raise ValueError(f"bootstrap log_slots must be positive, got {log_slots}")

        level_budget = _normalize_pair(self.level_budget, "level_budget")
        if level_budget[0] <= 1 or level_budget[1] <= 1:
            raise NotImplementedError(
                "OpenFHE bootstrap does not support the linear-transform route; "
                f"both level_budget entries must be greater than 1, got {level_budget}"
            )
        if level_budget[0] > log_slots or level_budget[1] > log_slots:
            raise ValueError(
                "bootstrap level_budget entries must not exceed log_slots, "
                f"got level_budget={level_budget}, log_slots={log_slots}"
            )

        output_levels = int(self.output_levels)
        if output_levels < 0:
            raise ValueError(f"bootstrap output_levels must be non-negative, got {output_levels}")

        strategy = str(self.strategy)
        if strategy not in _BOOTSTRAP_STRATEGIES:
            raise ValueError(
                "bootstrap strategy must be one of: double_hoist, normal_giant, normal_bsgs"
            )

        mode = str(self.mode)
        if mode not in _BOOTSTRAP_MODES:
            raise ValueError("bootstrap mode must be one of: modraise_first, stc_first")

        if self.dim1 is not None and self.baby_step is not None:
            raise ValueError("BootstrapSpec accepts either dim1 or baby_step, not both")
        dim1 = None if self.dim1 is None else _normalize_pair(self.dim1, "dim1")
        baby_step = None if self.baby_step is None else _normalize_pair(self.baby_step, "baby_step")
        if dim1 is not None and any(value < 0 for value in dim1):
            raise ValueError(f"bootstrap dim1 entries must be non-negative, got {dim1}")
        if baby_step is not None and any(value <= 0 for value in baby_step):
            raise ValueError(f"bootstrap baby_step entries must be positive, got {baby_step}")

        raise_to_limbs = None if self.raise_to_limbs is None else int(self.raise_to_limbs)
        if raise_to_limbs is not None and raise_to_limbs <= 0:
            raise ValueError(
                f"bootstrap raise_to_limbs must be positive when provided, got {raise_to_limbs}"
            )

        object.__setattr__(self, "log_slots", log_slots)
        object.__setattr__(self, "level_budget", level_budget)
        object.__setattr__(self, "output_levels", output_levels)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "dim1", dim1)
        object.__setattr__(self, "baby_step", baby_step)
        object.__setattr__(self, "raise_to_limbs", raise_to_limbs)

    @property
    def slots(self):
        return 1 << self.log_slots


@dataclass(frozen=True)
class BootstrapRequirements:
    """Context capacity and key requirements derived from bootstrap specs."""

    bootstrap_depth: int
    context_depth: int
    rotations: tuple[int, ...]


@dataclass(frozen=True)
class BootstrapProgram:
    """Context-bound constants and runtime plan for one bootstrap spec."""

    spec: BootstrapSpec
    constants: object = field(repr=False)
    _runtime_plan: object = field(repr=False)
    context_fingerprint: tuple = field(repr=False)
    raise_to_limbs: int
    output_state: CipherState


def requirements(
    specs: BootstrapSpec | Sequence[BootstrapSpec],
    *,
    log_n: int,
    secret_key_dist: str = "SPARSE_TERNARY",
):
    """Combine context-depth and rotation-key requirements for one or more specs."""

    normalized_specs = _normalize_specs(specs)
    log_n = int(log_n)
    if log_n <= 1:
        raise ValueError(f"log_n must be greater than 1, got {log_n}")

    max_slots_log = log_n - 1
    for spec in normalized_specs:
        if spec.log_slots > max_slots_log:
            raise ValueError(
                f"bootstrap log_slots={spec.log_slots} exceeds the context maximum {max_slots_log}"
            )

    depths = tuple(
        bootstrap_depth(spec.log_slots, spec.level_budget, secret_key_dist)
        for spec in normalized_specs
    )
    rotations = _unique_preserve_order(
        rotation
        for spec in normalized_specs
        for rotation in required_rotations(
            log_n,
            spec.log_slots,
            spec.level_budget,
            strategy=spec.strategy,
            dim1=spec.dim1,
            baby_step=spec.baby_step,
        )
    )
    required_context_depths = []
    for depth, spec in zip(depths, normalized_specs):
        minimum_raise_to_limbs = depth + spec.output_levels + 1
        if spec.raise_to_limbs is not None and spec.raise_to_limbs < minimum_raise_to_limbs:
            raise ValueError(
                "bootstrap raise_to_limbs is too small for the requested output: "
                f"need at least {minimum_raise_to_limbs}, got {spec.raise_to_limbs}"
            )
        required_context_depths.append(
            max(
                depth + spec.output_levels,
                0 if spec.raise_to_limbs is None else spec.raise_to_limbs - 1,
            )
        )

    return BootstrapRequirements(
        bootstrap_depth=max(depths),
        context_depth=max(required_context_depths),
        rotations=rotations,
    )


def _normalize_specs(specs):
    if isinstance(specs, BootstrapSpec):
        return (specs,)
    try:
        normalized = tuple(specs)
    except TypeError as exc:
        raise TypeError("requirements expects a BootstrapSpec or a sequence of BootstrapSpec") from exc
    if not normalized:
        raise ValueError("requirements expects at least one BootstrapSpec")
    if not all(isinstance(spec, BootstrapSpec) for spec in normalized):
        raise TypeError("requirements expects only BootstrapSpec values")
    return normalized


def _normalize_pair(value, name):
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"bootstrap {name} must have two entries, got {value}")
    try:
        if len(value) != 2:
            raise ValueError
        result = (int(value[0]), int(value[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"bootstrap {name} must have two integer entries, got {value}") from exc
    return result


def _unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
