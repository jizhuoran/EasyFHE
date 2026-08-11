from __future__ import annotations

from easyfhe.fhe.ciphertext import CipherState

from .generation.constants import generate_bootstrap_constants
from .generation.plan import describe_flat_ps_plan
from .generation.requirements import bootstrap_depth as _bootstrap_depth
from .generation.requirements import required_rotations as _required_rotations
from .spec import BootstrapProgram, BootstrapSpec


def generate(crypto_context, spec: BootstrapSpec):
    """Generate a context-bound bootstrap program from a public spec."""

    if not isinstance(spec, BootstrapSpec):
        raise TypeError("bootstrap generate expects a BootstrapSpec")

    max_limbs = int(crypto_context.max_limbs)
    max_slots = int(crypto_context.max_slots)
    if spec.slots > max_slots:
        raise ValueError(
            f"bootstrap slots={spec.slots} exceeds the context maximum {max_slots}"
        )

    raise_to_limbs = max_limbs if spec.raise_to_limbs is None else spec.raise_to_limbs
    if raise_to_limbs > max_limbs:
        raise ValueError(
            f"bootstrap raise_to_limbs={raise_to_limbs} exceeds context max_limbs={max_limbs}"
        )

    output_limbs = spec.output_levels + 1
    if output_limbs > raise_to_limbs:
        raise ValueError(
            "bootstrap output does not fit below its raise target: "
            f"output_limbs={output_limbs}, raise_to_limbs={raise_to_limbs}"
        )
    required_depth = _bootstrap_depth(
        spec.log_slots,
        spec.level_budget,
        crypto_context.params.secret_key_dist,
    )
    available_depth = raise_to_limbs - output_limbs
    if available_depth < required_depth:
        raise ValueError(
            "bootstrap raise target does not provide enough execution depth: "
            f"need {required_depth}, have {available_depth}"
        )

    planned_rotations = _required_rotations(
        crypto_context.params.log_n,
        spec.log_slots,
        spec.level_budget,
        strategy=spec.strategy,
        dim1=spec.dim1,
        baby_step=spec.baby_step,
    )
    _validate_rotation_keys(crypto_context, planned_rotations)

    constants, runtime_plan = generate_bootstrap_constants(
        crypto_context,
        log_slots=spec.log_slots,
        level_budget=spec.level_budget,
        output_levels=spec.output_levels,
        raise_to_limbs=raise_to_limbs,
        dim1=spec.dim1,
        baby_step=spec.baby_step,
        strategy=spec.strategy,
    )
    return BootstrapProgram(
        spec=spec,
        constants=constants,
        _runtime_plan=runtime_plan,
        context_fingerprint=_context_fingerprint(crypto_context),
        raise_to_limbs=raise_to_limbs,
        output_state=CipherState(
            cur_limbs=output_limbs,
            noise_deg=1,
            scaling_factor=float(crypto_context.scale_at(raise_to_limbs)),
        ),
    )


def describe_plan(program: BootstrapProgram):
    if not isinstance(program, BootstrapProgram):
        raise TypeError("describe_plan expects a BootstrapProgram")
    plan = program._runtime_plan
    return describe_flat_ps_plan(plan.approx_eval_plan, plan)


def bootstrap(cipher, crypto_context, program: BootstrapProgram):
    """Execute a context-bound bootstrap program."""

    if not isinstance(program, BootstrapProgram):
        raise TypeError("bootstrap expects a BootstrapProgram returned by generate")
    if program.context_fingerprint != _context_fingerprint(crypto_context):
        raise ValueError("bootstrap program was generated for a different context")
    if int(cipher.state.cur_limbs) > int(program.raise_to_limbs):
        raise ValueError(
            "bootstrap input has more limbs than the program raise target; "
            f"input cur_limbs={cipher.state.cur_limbs}, "
            f"raise_to_limbs={program.raise_to_limbs}"
        )
    if int(cipher.slots) > int(program.spec.slots):
        raise ValueError(
            f"bootstrap input slots={cipher.slots} exceeds program slots={program.spec.slots}"
        )

    from .runtime import homo_bootstrap

    return homo_bootstrap(
        cipher,
        crypto_context,
        program.constants,
        program._runtime_plan,
        raise_to_limbs=program.raise_to_limbs,
        output_state=program.output_state,
        mode=program.spec.mode,
    )


def _context_fingerprint(crypto_context):
    params = crypto_context.params
    return (
        int(crypto_context.ring_dim),
        int(params.dnum),
        str(params.scale_mode),
        str(params.secret_key_dist),
        params.q_primes,
        params.p_primes,
    )


def _validate_rotation_keys(crypto_context, required):
    available = getattr(crypto_context, "left_rot_key_map", None)
    if available is None:
        return
    missing = tuple(int(rotation) for rotation in required if int(rotation) not in available)
    if missing:
        preview = ", ".join(str(rotation) for rotation in missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(
            "context is missing bootstrap rotation keys: " + preview + suffix
        )
