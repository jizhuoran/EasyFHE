from __future__ import annotations

from dataclasses import dataclass

from ..bootstrap.approx_plan import bootstrap_approx_depth
from ..context import Context


@dataclass(frozen=True)
class BootstrapSpec:
    log_bs_slots: int
    level_budget: tuple[int, int]

    def __post_init__(self):
        object.__setattr__(self, "log_bs_slots", int(self.log_bs_slots))
        if len(self.level_budget) != 2:
            raise ValueError(f"bootstrap level_budget must have two entries, got {self.level_budget}")
        object.__setattr__(
            self,
            "level_budget",
            (int(self.level_budget[0]), int(self.level_budget[1])),
        )

    def budget_list(self):
        return [int(self.level_budget[0]), int(self.level_budget[1])]


@dataclass(frozen=True)
class CKKSContextSpec:
    depth: int
    log_n: int
    dnum: int
    dcrt_bits: int
    first_mod: int
    secret_key_dist: str = "SPARSE_TERNARY"
    rescale_tech: str = "FIXEDMANUAL"
    rotations: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "log_n", int(self.log_n))
        object.__setattr__(self, "dnum", int(self.dnum))
        object.__setattr__(self, "dcrt_bits", int(self.dcrt_bits))
        object.__setattr__(self, "first_mod", int(self.first_mod))
        object.__setattr__(self, "secret_key_dist", str(self.secret_key_dist))
        object.__setattr__(self, "rescale_tech", str(self.rescale_tech))
        object.__setattr__(self, "rotations", tuple(int(rotation) for rotation in (self.rotations or ())))


def _normalize_bootstrap_specs(bootstrap_specs):
    return tuple(
        spec if isinstance(spec, BootstrapSpec) else BootstrapSpec(spec[0], spec[1])
        for spec in (bootstrap_specs or ())
    )


def bootstrap_depth(max_levels_remaining, bootstrap_specs, secret_key_dist="SPARSE_TERNARY"):
    """Return the CKKS multiplicative depth needed when bootstrap is used."""

    bootstrap_specs = _normalize_bootstrap_specs(bootstrap_specs)
    if not bootstrap_specs:
        return int(max_levels_remaining)
    secret_key_dist = str(secret_key_dist)
    max_budget = max(bootstrap_specs, key=lambda spec: sum(spec.level_budget)).level_budget
    approx_mod_depth = bootstrap_approx_depth(secret_key_dist)
    return int(max_levels_remaining) + approx_mod_depth + int(max_budget[0]) + int(max_budget[1])


def generate_context(spec: CKKSContextSpec, device="cpu", options=None):
    if not isinstance(spec, CKKSContextSpec):
        raise TypeError("generate_context expects a CKKSContextSpec")
    return Context.build(spec, device, options)
