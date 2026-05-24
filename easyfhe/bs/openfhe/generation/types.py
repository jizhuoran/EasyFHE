from __future__ import annotations

from dataclasses import dataclass


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
    baby_step: tuple[int, int]
    strategy: str
    post_bootstrap_levels: int
    c2s_plan: BootstrapTransformPlan
    s2c_plan: BootstrapTransformPlan
    approx_eval_plan: object
    approx_tail_scalar_names: tuple[tuple[str, ...], ...]
    approx_constant_scalar_names: dict[tuple[str, ...], str]
    approx_q_highest_scalar_names: dict[tuple[str, ...], str]
    chebyshev_neg_one_scalar_name: str
    double_angle_iterations: int
    double_angle_scalar_names: tuple[str, ...]
    required_rotations: tuple[int, ...]

    @property
    def slots(self):
        return 1 << self.log_bs_slots

    @property
    def max_levels_remaining(self):
        return self.post_bootstrap_levels

    def describe_approx(self):
        from .plan import describe_flat_ps_plan

        return describe_flat_ps_plan(self.approx_eval_plan, self)
