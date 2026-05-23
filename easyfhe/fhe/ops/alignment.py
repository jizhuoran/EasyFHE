from ..ciphertext import CipherState
from . import kernels as F


def plan_add_alignment(in0, in1, context) -> CipherState:
    if context.rescale_policy == "manual":
        if in0.state.noise_deg != in1.state.noise_deg:
            raise ValueError(
                f"plan_add_alignment: noise_deg mismatch: {in0.state.noise_deg} != {in1.state.noise_deg}"
            )
        return CipherState(
            cur_limbs=min(in0.state.cur_limbs, in1.state.cur_limbs),
            noise_deg=in0.state.noise_deg,
            scaling_factor=None,
        )
    return _plan_auto_pair_alignment(in0, in1)


def plan_mul_alignment(in0, in1, context) -> tuple[CipherState, CipherState]:
    if context.rescale_policy == "manual":
        target_limbs = min(in0.state.cur_limbs, in1.state.cur_limbs)
        return (
            CipherState(target_limbs, in0.state.noise_deg, None),
            CipherState(target_limbs, in1.state.noise_deg, None),
        )
    target = _plan_auto_pair_alignment(in0, in1)
    target = _mul_ready_target(target, context)
    return target, target


def align_to(cipher, target: CipherState, context):
    _validate_alignment_request(cipher, target)
    if context.scale_mode == "fixed":
        return _align_fixed(cipher, target, context)
    if context.scale_mode == "flexible":
        return _align_flexible(cipher, target, context)
    raise ValueError(f"Unsupported scale mode: {context.scale_mode}")


def reduce_noise_to_one(cipher, context):
    return align_to(cipher, _mul_ready_target(cipher.state, context), context)


def rescale_one_level(cipher, context):
    if cipher.is_ext:
        raise ValueError("rescale_one_level: ext ciphers must be moddowned before rescale")
    if cipher.state.cur_limbs <= 1:
        raise ValueError(f"rescale_one_level: cur_limbs must be > 1, got {cipher.state.cur_limbs}")
    if cipher.state.noise_deg <= 1:
        raise ValueError(f"rescale_one_level: noise_deg must be > 1, got {cipher.state.noise_deg}")
    return align_to(cipher, CipherState(cipher.state.cur_limbs - 1, cipher.state.noise_deg - 1), context)


# Common planning and state helpers.


def _coerce_state(cipher_or_state) -> CipherState:
    if isinstance(cipher_or_state, CipherState):
        return cipher_or_state
    return cipher_or_state.state


def _consumed_depth(cipher_or_state) -> int:
    state = _coerce_state(cipher_or_state)
    return state.cur_limbs - state.noise_deg


def _validate_alignment_request(cipher, target: CipherState):
    if _consumed_depth(cipher) < _consumed_depth(target):
        raise ValueError(
            "align_to: target state consumes more depth than source state: "
            f"source cur_limbs={cipher.state.cur_limbs}, source noise_deg={cipher.state.noise_deg}, "
            f"target cur_limbs={target.cur_limbs}, target noise_deg={target.noise_deg}"
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
    return CipherState(in0.state.cur_limbs, max(in0.state.noise_deg, in1.state.noise_deg), None)


def _mul_ready_target(target: CipherState, context) -> CipherState:
    if target.noise_deg == 1:
        return target
    if target.noise_deg != 2:
        raise ValueError(f"Unsupported multiplication input noise degree: {target.noise_deg}")

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
    if cipher.state.noise_deg <= 1:
        raise ValueError(f"_rescale_one_level: noise_deg must be > 1, got {cipher.state.noise_deg}")

    res_cv = [F.cv_rescale_one_level(cv, cipher.state.cur_limbs, 0, context) for cv in cipher.cv]
    divisor = context.rescale_divisor_at(cipher.state.cur_limbs - 1)
    return cipher.cipher_like(
        res_cv,
        state=CipherState(
            cipher.state.cur_limbs - 1,
            cipher.state.noise_deg - 1,
            cipher.state.scaling_factor / divisor,
        ),
    )

# Fixed-scale alignment.


def _align_fixed(cipher, target, context):
    if cipher.state.noise_deg < target.noise_deg:
        raise ValueError(
            "_align_fixed: cannot increase noise_deg: "
            f"{cipher.state.noise_deg} -> {target.noise_deg}"
        )

    while cipher.state.noise_deg > target.noise_deg:
        cipher = _rescale_one_level(cipher, context)

    if cipher.state.cur_limbs > target.cur_limbs:
        cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return cipher.shallow_copy()


# Flexible-scale alignment.


def _align_flexible(cipher, target, context):
    if cipher.state.cur_limbs == target.cur_limbs:
        return _align_flexible_same_limbs(cipher, target, context)

    transition = (cipher.state.noise_deg, target.noise_deg)
    if transition == (2, 2):
        return _align_flexible_2_to_2(cipher, target, context)
    if transition == (1, 1):
        return _align_flexible_1_to_1(cipher, target, context)
    if transition == (2, 1):
        return _align_flexible_2_to_1(cipher, target, context)
    if transition == (1, 2):
        return _align_flexible_1_to_2(cipher, target, context)
    raise ValueError(f"_align_flexible: unsupported noise transition {transition[0]} -> {transition[1]}")


def _align_flexible_same_limbs(cipher, target, context):
    if cipher.state.noise_deg < target.noise_deg:
        return _multiply_by_scale_correction(cipher, 1.0, context)
    return cipher.shallow_copy()


def _align_flexible_2_to_2(cipher, target, context):
    target_scale = _require_target_scale(target, "_align_flexible_2_to_2")
    scale = context.scale_at(cipher.state.cur_limbs)
    divisor = context.rescale_divisor_at(cipher.state.cur_limbs - 1)
    correction_factor = target_scale / cipher.state.scaling_factor * divisor / scale
    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    cipher = _rescale_one_level(cipher, context)
    if cipher.state.cur_limbs > target.cur_limbs:
        cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return _with_target_scale(cipher, target)


def _align_flexible_1_to_1(cipher, target, context):
    correction_factor = _flexible_1_to_1_correction_factor(cipher, target, context)
    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    if cipher.state.cur_limbs > target.cur_limbs + 1:
        cipher = _drop_to_limbs(cipher, target.cur_limbs + 1, context)
    cipher = _rescale_one_level(cipher, context)
    return _with_target_scale(cipher, target)


def _align_flexible_2_to_1(cipher, target, context):
    if cipher.state.cur_limbs == target.cur_limbs + 1:
        cipher = _rescale_one_level(cipher, context)
        return _with_target_scale(cipher, target)

    pre_rescale_target_scale = _pre_rescale_target_scale(target, context)
    scale = context.scale_at(cipher.state.cur_limbs)
    divisor = context.rescale_divisor_at(cipher.state.cur_limbs - 1)
    correction_factor = pre_rescale_target_scale / cipher.state.scaling_factor * divisor / scale
    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    cipher = _rescale_one_level(cipher, context)
    if cipher.state.cur_limbs > target.cur_limbs + 1:
        cipher = _drop_to_limbs(cipher, target.cur_limbs + 1, context)
    cipher = _rescale_one_level(cipher, context)
    return _with_target_scale(cipher, target)


def _align_flexible_1_to_2(cipher, target, context):
    target_scale = _require_target_scale(target, "_align_flexible_1_to_2")
    correction_factor = target_scale / cipher.state.scaling_factor / context.scale_at(cipher.state.cur_limbs)
    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return cipher.cipher_like(cipher.cv, state=cipher.state.replace(scaling_factor=target_scale))


def _flexible_1_to_1_correction_factor(cipher, target, context):
    if not _has_target_scale(target):
        return 1.0
    return (
        target.scaling_factor
        * context.rescale_divisor_at(target.cur_limbs)
        / cipher.state.scaling_factor
        / context.scale_at(cipher.state.cur_limbs)
    )


def _multiply_by_scale_correction(cipher, correction_factor, context):
    if cipher.is_ext:
        raise ValueError("_multiply_by_scale_correction: ext ciphers must be moddowned before scale correction")

    factors = _scale_correction_factors(correction_factor, cipher.state.cur_limbs, context)
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
    scale = context.scale_at(cipher.state.cur_limbs)
    return cipher.cipher_like(
        cv,
        state=CipherState(
            cipher.state.cur_limbs,
            cipher.state.noise_deg + 1,
            cipher.state.scaling_factor * scale,
        ),
    )


def _scale_correction_factors(correction_factor, cur_limbs, context):
    scale = context.scale_at(cur_limbs)
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
