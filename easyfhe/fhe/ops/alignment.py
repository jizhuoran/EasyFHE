from ..ciphertext import CipherState
from . import kernels as F


def plan_add_alignment(in0, in1, context) -> CipherState:
    if context.scale_mode == "flexible":
        raise ValueError("flexible mode does not support implicit add alignment; call align_to explicitly")
    if context.rescale_policy == "manual":
        if in0.state.scale_degree != in1.state.scale_degree:
            raise ValueError(
                f"plan_add_alignment: scale_degree mismatch: {in0.state.scale_degree} != {in1.state.scale_degree}"
            )
        return CipherState(
            cur_limbs=min(in0.state.cur_limbs, in1.state.cur_limbs),
            scale_degree=in0.state.scale_degree,
            scaling_factor=None,
        )
    return _plan_auto_pair_alignment(in0, in1)


def plan_mul_alignment(in0, in1, context) -> tuple[CipherState, CipherState]:
    if context.scale_mode == "flexible":
        raise ValueError("flexible mode does not support implicit multiply alignment; call align_to explicitly")
    if context.rescale_policy == "manual":
        target_limbs = min(in0.state.cur_limbs, in1.state.cur_limbs)
        return (
            CipherState(target_limbs, in0.state.scale_degree, None),
            CipherState(target_limbs, in1.state.scale_degree, None),
        )
    target = _plan_auto_pair_alignment(in0, in1)
    target = _mul_ready_target(target, context)
    return target, target


def align_to(cipher, target: CipherState, context):
    _validate_alignment_request(cipher, target)
    if context.scale_mode == "fixed":
        return _align_fixed(cipher, target, context)
    if context.scale_mode == "flexible":
        if not _has_target_scale(target):
            raise ValueError("flexible align_to requires target.scaling_factor")
        return _align_flexible(cipher, target, context)
    raise ValueError(f"Unsupported scale mode: {context.scale_mode}")


def normalize_scale(cipher, context):
    return align_to(cipher, _mul_ready_target(cipher.state, context), context)


def rescale(cipher, context):
    if cipher.is_ext:
        raise ValueError("rescale: ext ciphers must be moddowned before rescale")
    if cipher.state.cur_limbs <= 1:
        raise ValueError(f"rescale: cur_limbs must be > 1, got {cipher.state.cur_limbs}")
    if cipher.state.scale_degree <= 1:
        raise ValueError(f"rescale: scale_degree must be > 1, got {cipher.state.scale_degree}")
    target_scale = None
    if context.scale_mode == "flexible" and cipher.state.scaling_factor is not None:
        target_scale = float(cipher.state.scaling_factor) / float(context.rescale_divisor_at(cipher.state.cur_limbs - 1))
    return align_to(
        cipher,
        CipherState(cipher.state.cur_limbs - 1, cipher.state.scale_degree - 1, target_scale),
        context,
    )


# Common planning and state helpers.


def _coerce_state(cipher_or_state) -> CipherState:
    if isinstance(cipher_or_state, CipherState):
        return cipher_or_state
    return cipher_or_state.state


def _consumed_depth(cipher_or_state) -> int:
    state = _coerce_state(cipher_or_state)
    return state.cur_limbs - state.scale_degree


def _validate_alignment_request(cipher, target: CipherState):
    if _consumed_depth(cipher) < _consumed_depth(target):
        raise ValueError(
            "align_to: target state consumes more depth than source state: "
            f"source cur_limbs={cipher.state.cur_limbs}, source scale_degree={cipher.state.scale_degree}, "
            f"target cur_limbs={target.cur_limbs}, target scale_degree={target.scale_degree}"
        )
    if cipher.state.cur_limbs < target.cur_limbs:
        raise ValueError(
            "align_to: target cur_limbs exceeds source cur_limbs: "
            f"{target.cur_limbs} > {cipher.state.cur_limbs}"
        )


def _plan_auto_pair_alignment(in0, in1) -> CipherState:
    if in0.state.cur_limbs > in1.state.cur_limbs:
        return in1.state
    if in0.state.cur_limbs < in1.state.cur_limbs:
        return in0.state
    return CipherState(in0.state.cur_limbs, max(in0.state.scale_degree, in1.state.scale_degree), None)


def _mul_ready_target(target: CipherState, context) -> CipherState:
    if target.scale_degree == 1:
        return target
    if target.scale_degree != 2:
        raise ValueError(f"Unsupported multiplication input scale degree: {target.scale_degree}")

    scaling_factor = target.scaling_factor
    if scaling_factor is not None:
        scaling_factor = scaling_factor / context.rescale_divisor_at(target.cur_limbs - 1)
    return CipherState(target.cur_limbs - 1, 1, scaling_factor)


# Common ciphertext transforms.


def _drop_to_limbs(cipher, target_limbs, context):
    if cipher.is_ext:
        raise ValueError("_drop_to_limbs: ext ciphers must be moddowned before dropping limbs")
    if target_limbs < 0 or target_limbs > cipher.state.cur_limbs:
        raise ValueError(
            f"_drop_to_limbs: target_limbs must be between 0 and cur_limbs, got "
            f"target_limbs={target_limbs}, cur_limbs={cipher.state.cur_limbs}"
        )
    target_limbs = int(target_limbs)
    if target_limbs == cipher.state.cur_limbs:
        return cipher.shallow_copy()
    return cipher.cipher_like(cipher.cv, state=cipher.state.replace(cur_limbs=target_limbs))


def _rescale_one_level(cipher, context):
    if cipher.is_ext:
        raise ValueError("_rescale_one_level: ext ciphers must be moddowned before rescale")
    if cipher.state.cur_limbs <= 1:
        raise ValueError(f"_rescale_one_level: cur_limbs must be > 1, got {cipher.state.cur_limbs}")
    if cipher.state.scale_degree <= 1:
        raise ValueError(f"_rescale_one_level: scale_degree must be > 1, got {cipher.state.scale_degree}")

    res_cv = [F.cv_rescale_one_level(cv, cipher.state.cur_limbs, 0, context) for cv in cipher.cv]
    divisor = context.rescale_divisor_at(cipher.state.cur_limbs - 1)
    return cipher.cipher_like(
        res_cv,
        state=CipherState(
            cipher.state.cur_limbs - 1,
            cipher.state.scale_degree - 1,
            cipher.state.scaling_factor / divisor,
        ),
    )

# Fixed-scale alignment.


def _align_fixed(cipher, target, context):
    if cipher.state.scale_degree < target.scale_degree:
        raise ValueError(
            "_align_fixed: cannot increase scale_degree: "
            f"{cipher.state.scale_degree} -> {target.scale_degree}"
        )

    while cipher.state.scale_degree > target.scale_degree:
        cipher = _rescale_one_level(cipher, context)

    if cipher.state.cur_limbs > target.cur_limbs:
        cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return cipher.shallow_copy()


# Flexible-scale alignment.


def _align_flexible(cipher, target, context):
    _validate_flexible_scale_degree(cipher.state.scale_degree, "_align_flexible source")
    _validate_flexible_scale_degree(target.scale_degree, "_align_flexible target")
    if cipher.state.cur_limbs == target.cur_limbs:
        return _align_flexible_same_limbs(cipher, target, context)

    transition = (cipher.state.scale_degree, target.scale_degree)
    if transition == (2, 2):
        return _align_flexible_2_to_2(cipher, target, context)
    if transition == (1, 1):
        return _align_flexible_1_to_1(cipher, target, context)
    if transition == (2, 1):
        return _align_flexible_2_to_1(cipher, target, context)
    if transition == (1, 2):
        return _align_flexible_1_to_2(cipher, target, context)
    raise ValueError(f"_align_flexible: unsupported scale-degree transition {transition[0]} -> {transition[1]}")


def _align_flexible_same_limbs(cipher, target, context):
    if cipher.state.scale_degree == target.scale_degree:
        if _has_target_scale(target) and not _scale_close(cipher.state.scaling_factor, target.scaling_factor):
            raise ValueError(
                "_align_flexible cannot change scale at the same limb/scale-degree state "
                "without consuming a rescale limb: "
                f"cur_limbs={cipher.state.cur_limbs}, scale_degree={cipher.state.scale_degree}"
            )
        return _with_target_scale(cipher.shallow_copy(), target)
    if cipher.state.scale_degree == 1 and target.scale_degree == 2:
        return _align_flexible_1_to_2(cipher, target, context)
    if cipher.state.scale_degree == 2 and target.scale_degree == 1:
        raise ValueError(
            "_align_flexible cannot reduce scale_degree at the same limb count; "
            "one rescale limb must be consumed"
        )
    return cipher.shallow_copy()


def _align_flexible_2_to_2(cipher, target, context):
    if _scale_close(cipher.state.scaling_factor, target.scaling_factor):
        if cipher.state.cur_limbs == target.cur_limbs:
            return _with_target_scale(cipher, target)
    if cipher.state.cur_limbs <= target.cur_limbs:
        raise ValueError(
            "_align_flexible_2_to_2 cannot change scale without first rescaling to scale_degree=1"
        )
    target_scale = _require_target_scale(target, "_align_flexible_2_to_2")
    divisor = context.rescale_divisor_at(cipher.state.cur_limbs - 1)
    correction_factor = (
        target_scale
        / _require_scale(cipher.state.scaling_factor, "_align_flexible_2_to_2")
        * divisor
        / target_scale
    )
    cipher = _multiply_by_scale_correction(
        cipher,
        correction_factor,
        context,
        scaling_factor=target_scale,
    )
    cipher = _rescale_one_level(cipher, context)
    if cipher.state.cur_limbs > target.cur_limbs:
        cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return _with_target_scale(cipher, target)


def _align_flexible_1_to_1(cipher, target, context):
    if _scale_close(cipher.state.scaling_factor, target.scaling_factor):
        if cipher.state.cur_limbs == target.cur_limbs:
            return _with_target_scale(cipher, target)
    if cipher.state.cur_limbs <= target.cur_limbs:
        raise ValueError(
            "_align_flexible_1_to_1 cannot change scale without consuming one rescale limb"
        )
    pre_rescale_limbs = int(target.cur_limbs) + 1
    if cipher.state.cur_limbs > pre_rescale_limbs:
        cipher = _drop_to_limbs(cipher, pre_rescale_limbs, context)
    divisor = context.rescale_divisor_at(target.cur_limbs)
    correction_scale = float(target.scaling_factor) * float(divisor) / _require_scale(
        cipher.state.scaling_factor,
        "_align_flexible_1_to_1",
    )
    cipher = _multiply_by_scale_correction(cipher, 1.0, context, scaling_factor=correction_scale)
    cipher = _rescale_one_level(cipher, context)
    return _with_target_scale(cipher, target)


def _align_flexible_2_to_1(cipher, target, context):
    if cipher.state.cur_limbs == target.cur_limbs + 1:
        cipher = _rescale_one_level(cipher, context)
        if _has_target_scale(target) and not _scale_close(cipher.state.scaling_factor, target.scaling_factor):
            raise ValueError(
                "_align_flexible_2_to_1 target scale is not reachable at the requested limb count: "
                f"natural_scale={cipher.state.scaling_factor}, target_scale={target.scaling_factor}"
            )
        return _with_target_scale(cipher, target)
    pre_rescale_target_scale = _pre_rescale_target_scale(target, context)
    divisor = context.rescale_divisor_at(cipher.state.cur_limbs - 1)
    correction_factor = (
        pre_rescale_target_scale
        / _require_scale(cipher.state.scaling_factor, "_align_flexible_2_to_1")
        * divisor
        / pre_rescale_target_scale
    )
    cipher = _multiply_by_scale_correction(
        cipher,
        correction_factor,
        context,
        scaling_factor=pre_rescale_target_scale,
    )
    cipher = _rescale_one_level(cipher, context)
    pre_rescale_limbs = int(target.cur_limbs) + 1
    if cipher.state.cur_limbs > pre_rescale_limbs:
        cipher = _drop_to_limbs(cipher, pre_rescale_limbs, context)
    cipher = _rescale_one_level(cipher, context)
    return _with_target_scale(cipher, target)


def _align_flexible_1_to_2(cipher, target, context):
    if cipher.state.cur_limbs > target.cur_limbs:
        cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    correction_scale = float(target.scaling_factor) / _require_scale(
        cipher.state.scaling_factor,
        "_align_flexible_1_to_2",
    )
    cipher = _multiply_by_scale_correction(cipher, 1.0, context, scaling_factor=correction_scale)
    return _with_target_scale(cipher, target)


def _multiply_by_scale_correction(cipher, correction_factor, context, *, scaling_factor=None):
    if cipher.is_ext:
        raise ValueError("_multiply_by_scale_correction: ext ciphers must be moddowned before scale correction")

    scale = context.scale_at(cipher.state.cur_limbs) if scaling_factor is None else float(scaling_factor)
    factors = _scale_correction_factors(correction_factor, cipher.state.cur_limbs, context, scaling_factor=scale)
    scalar_mod = F.gen_scalar_tensor(factors, context.moduliQ_scalar, cipher.state.cur_limbs)
    scalar_mod = scalar_mod.to(cipher.cv[0].device)
    cv = [
        F.cv_mul_scalar(
            component,
            scalar_mod,
            context.moduliQ,
            context.q_mu,
            cipher.state.cur_limbs,
        )
        for component in cipher.cv
    ]
    return cipher.cipher_like(
        cv,
        state=CipherState(
            cipher.state.cur_limbs,
            cipher.state.scale_degree + 1,
            cipher.state.scaling_factor * scale,
        ),
    )


def _scale_correction_factors(correction_factor, cur_limbs, context, *, scaling_factor=None):
    scale = context.scale_at(cur_limbs) if scaling_factor is None else float(scaling_factor)
    value = int(correction_factor * scale + 0.5)
    return [value % int(context.moduliQ_scalar[i]) for i in range(cur_limbs)]


def _has_target_scale(target: CipherState):
    return target.scaling_factor is not None


def _with_target_scale(cipher, target: CipherState):
    if not _has_target_scale(target):
        return cipher
    return cipher.cipher_like(cipher.cv, state=cipher.state.replace(scaling_factor=target.scaling_factor))


def _require_target_scale(target: CipherState, op_name: str):
    if not _has_target_scale(target):
        raise ValueError(f"{op_name}: target scaling_factor is required")
    return target.scaling_factor


def _pre_rescale_target_scale(target: CipherState, context):
    if _has_target_scale(target):
        return target.scaling_factor * context.rescale_divisor_at(target.cur_limbs)
    return context.big_scale_at(target.cur_limbs + 1)


def _validate_flexible_scale_degree(scale_degree, op_name):
    if int(scale_degree) not in (1, 2):
        raise ValueError(f"{op_name}: u64 flexible alignment supports only scale_degree 1 or 2, got {scale_degree}")


def _require_scale(scale, op_name):
    if scale is None:
        raise ValueError(f"{op_name}: scaling_factor is required for flexible scale alignment")
    return float(scale)


def _scale_close(actual, expected):
    actual = _require_scale(actual, "_scale_close actual")
    expected = _require_scale(expected, "_scale_close expected")
    if actual <= 0 or expected <= 0:
        return False
    rel = abs(actual - expected) / max(abs(actual), abs(expected))
    return rel <= 1e-7
