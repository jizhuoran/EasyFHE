from dataclasses import dataclass
from typing import Optional

from . import kernels as F


@dataclass(frozen=True)
class CipherState:
    cur_limbs: int
    noise_deg: int
    scaling_factor: Optional[float] = None


def state_of(cipher) -> CipherState:
    return CipherState(
        cur_limbs=int(cipher.cur_limbs),
        noise_deg=int(cipher.noise_deg),
        scaling_factor=cipher.scaling_factor,
    )


def consumed_depth(state_or_cipher) -> int:
    state = _coerce_state(state_or_cipher)
    return state.cur_limbs - state.noise_deg


def has_target_scale(target: CipherState) -> bool:
    return target.scaling_factor is not None


def plan_add_alignment(in0, in1, context) -> CipherState:
    if context.rescale_policy == "manual":
        if in0.noise_deg != in1.noise_deg:
            raise ValueError(
                f"plan_add_alignment: noise_deg mismatch: {in0.noise_deg} != {in1.noise_deg}"
            )
        return CipherState(
            cur_limbs=min(in0.cur_limbs, in1.cur_limbs),
            noise_deg=in0.noise_deg,
            scaling_factor=None,
        )
    return _plan_auto_pair_alignment(in0, in1)


def plan_mul_alignment(in0, in1, context) -> tuple[CipherState, CipherState]:
    if context.rescale_policy == "manual":
        target_limbs = min(in0.cur_limbs, in1.cur_limbs)
        return (
            CipherState(target_limbs, in0.noise_deg, None),
            CipherState(target_limbs, in1.noise_deg, None),
        )
    target = _plan_auto_pair_alignment(in0, in1)
    target = plan_reduce_noise_to_one(target, context)
    return target, target


def plan_reduce_noise_to_one(cipher_or_state, context) -> CipherState:
    state = _coerce_state(cipher_or_state)
    if context.rescale_policy == "manual":
        return state
    return _mul_ready_target(state, context)


def align_to(cipher, target: CipherState, context):
    return _align_to(cipher, target, context)


def reduce_noise_to_one(cipher, context):
    return align_to(cipher, _mul_ready_target(state_of(cipher), context), context)


def rescale_one_level(cipher, context):
    if cipher.cur_limbs <= 1:
        raise ValueError(f"rescale_one_level: cur_limbs must be > 1, got {cipher.cur_limbs}")
    if cipher.noise_deg <= 1:
        raise ValueError(f"rescale_one_level: noise_deg must be > 1, got {cipher.noise_deg}")
    return align_to(cipher, CipherState(cipher.cur_limbs - 1, cipher.noise_deg - 1), context)


def _align_to(cipher, target: CipherState, context):
    _validate_alignment_request(cipher, target)
    if context.scale_mode == "fixed":
        return _align_fixed(cipher, target, context)
    if context.scale_mode == "flexible":
        return _align_flexible(cipher, target, context)
    raise ValueError(f"Unsupported scale mode: {context.scale_mode}")


def _coerce_state(cipher_or_state) -> CipherState:
    if isinstance(cipher_or_state, CipherState):
        return cipher_or_state
    return state_of(cipher_or_state)


def _validate_alignment_request(cipher, target: CipherState):
    if consumed_depth(cipher) < consumed_depth(target):
        raise ValueError(
            "align_to: target state consumes more depth than source state: "
            f"source cur_limbs={cipher.cur_limbs}, source noise_deg={cipher.noise_deg}, "
            f"target cur_limbs={target.cur_limbs}, target noise_deg={target.noise_deg}"
        )
    if cipher.cur_limbs < target.cur_limbs:
        raise ValueError(
            "align_to: target cur_limbs exceeds source cur_limbs: "
            f"{target.cur_limbs} > {cipher.cur_limbs}"
        )


def _plan_auto_pair_alignment(in0, in1) -> CipherState:
    if in0.cur_limbs > in1.cur_limbs:
        return CipherState(in1.cur_limbs, in1.noise_deg, in1.scaling_factor)
    if in0.cur_limbs < in1.cur_limbs:
        return CipherState(in0.cur_limbs, in0.noise_deg, in0.scaling_factor)
    return CipherState(in0.cur_limbs, max(in0.noise_deg, in1.noise_deg), None)


def _mul_ready_target(target: CipherState, context) -> CipherState:
    if target.noise_deg == 1:
        return target
    if target.noise_deg != 2:
        raise ValueError(f"Unsupported multiplication input noise degree: {target.noise_deg}")

    scaling_factor = target.scaling_factor
    if scaling_factor is not None:
        scaling_factor = scaling_factor / context.rescale_divisor_at(target.cur_limbs - 1)
    return CipherState(target.cur_limbs - 1, 1, scaling_factor)


def _drop_to_limbs(cipher, target_limbs, context):
    if target_limbs < 0 or target_limbs > cipher.cur_limbs:
        raise ValueError(
            f"_drop_to_limbs: target_limbs must be between 0 and cur_limbs, got "
            f"target_limbs={target_limbs}, cur_limbs={cipher.cur_limbs}"
        )
    return cipher.cipher_like(cipher.cv, cur_limbs=target_limbs)


def _rescale_one_level(cipher, context):
    if cipher.cur_limbs <= 1:
        raise ValueError(f"_rescale_one_level: cur_limbs must be > 1, got {cipher.cur_limbs}")
    if cipher.noise_deg <= 1:
        raise ValueError(f"_rescale_one_level: noise_deg must be > 1, got {cipher.noise_deg}")
    res_cv = [
        F.cv_drop_last_element_and_scale(cv, cipher.cur_limbs, 0, context)
        for cv in cipher.cv
    ]
    mod_reduce_factor = context.rescale_divisor_at(cipher.cur_limbs - 1)
    return cipher.cipher_like(
        res_cv,
        cur_limbs=cipher.cur_limbs - 1,
        scaling_factor=cipher.scaling_factor / mod_reduce_factor,
        noise_deg=cipher.noise_deg - 1,
    )


def _scale_correction_factors(correction_factor, cur_limbs, context):
    sc_factor = context.scale_at(cur_limbs)
    value = int(correction_factor * sc_factor + 0.5)
    return [value % int(context.moduliQ_scalar[i]) for i in range(cur_limbs)]


def _multiply_by_scale_correction(cipher, correction_factor, context):
    factors = _scale_correction_factors(correction_factor, cipher.cur_limbs, context)
    scalar_mod = F.gen_scalar_tensor(factors, context.moduliQ_scalar, cipher.cur_limbs)
    scalar_mod = scalar_mod.to(cipher.cv[0].device)
    cv = [
        F.cv_mul_scalar(
            cv_i,
            scalar_mod,
            context.moduliQ,
            context.q_mu,
            cipher.cur_limbs,
        )
        for cv_i in cipher.cv
    ]
    sc_factor = context.scale_at(cipher.cur_limbs)
    return cipher.cipher_like(
        cv,
        scaling_factor=cipher.scaling_factor * sc_factor,
        noise_deg=cipher.noise_deg + 1,
    )


def _with_target_scale(cipher, target: CipherState):
    if not has_target_scale(target):
        return cipher
    return cipher.cipher_like(cipher.cv, scaling_factor=target.scaling_factor)


def _require_target_scale(target: CipherState, op_name: str):
    if not has_target_scale(target):
        raise ValueError(f"{op_name}: target scaling_factor is required")
    return target.scaling_factor


def _pre_rescale_target_scale(target: CipherState, context):
    if has_target_scale(target):
        return target.scaling_factor * context.rescale_divisor_at(target.cur_limbs)
    return context.big_scale_at(target.cur_limbs + 1)


def _align_fixed(cipher, target, context):
    if cipher.cur_limbs == target.cur_limbs:
        return _align_same_limbs_fixed(cipher, target, context)

    transition = (cipher.noise_deg, target.noise_deg)
    if transition == (2, 2):
        return _align_fixed_2_to_2(cipher, target, context)
    if transition == (1, 1):
        return _align_fixed_1_to_1(cipher, target, context)
    if transition == (2, 1):
        return _align_fixed_2_to_1(cipher, target, context)
    if transition == (1, 2):
        return _align_fixed_1_to_2(cipher, target, context)
    raise ValueError(f"_align_fixed: unsupported noise transition {transition[0]} -> {transition[1]}")


def _align_same_limbs_fixed(cipher, target, context):
    if cipher.noise_deg < target.noise_deg:
        return _multiply_by_scale_correction(cipher, 1.0, context)
    if cipher.noise_deg != target.noise_deg:
        raise ValueError(f"_align_fixed: noise_deg mismatch: {cipher.noise_deg} != {target.noise_deg}")
    return cipher.shallow_copy()


def _align_fixed_2_to_2(cipher, target, context):
    cipher = _multiply_by_scale_correction(cipher, 1.0, context)
    cipher = _rescale_one_level(cipher, context)
    if cipher.cur_limbs > target.cur_limbs:
        cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return cipher


def _align_fixed_1_to_1(cipher, target, context):
    cipher = _multiply_by_scale_correction(cipher, 1.0, context)
    if cipher.cur_limbs > target.cur_limbs + 1:
        cipher = _drop_to_limbs(cipher, target.cur_limbs + 1, context)
    return _rescale_one_level(cipher, context)


def _align_fixed_2_to_1(cipher, target, context):
    if cipher.cur_limbs == target.cur_limbs + 1:
        return _rescale_one_level(cipher, context)

    cipher = _multiply_by_scale_correction(cipher, 1.0, context)
    cipher = _rescale_one_level(cipher, context)
    if cipher.cur_limbs > target.cur_limbs + 1:
        cipher = _drop_to_limbs(cipher, target.cur_limbs + 1, context)
    return _rescale_one_level(cipher, context)


def _align_fixed_1_to_2(cipher, target, context):
    cipher = _multiply_by_scale_correction(cipher, 1.0, context)
    return _drop_to_limbs(cipher, target.cur_limbs, context)


def _align_flexible(cipher, target, context):
    if cipher.cur_limbs == target.cur_limbs:
        return _align_same_limbs_flexible(cipher, target, context)

    transition = (cipher.noise_deg, target.noise_deg)
    if transition == (2, 2):
        return _align_flexible_2_to_2(cipher, target, context)
    if transition == (1, 1):
        return _align_flexible_1_to_1(cipher, target, context)
    if transition == (2, 1):
        return _align_flexible_2_to_1(cipher, target, context)
    if transition == (1, 2):
        return _align_flexible_1_to_2(cipher, target, context)
    raise ValueError(f"_align_flexible: unsupported noise transition {transition[0]} -> {transition[1]}")


def _align_same_limbs_flexible(cipher, target, context):
    if cipher.noise_deg < target.noise_deg:
        return _multiply_by_scale_correction(cipher, 1.0, context)
    return cipher.shallow_copy()


def _align_flexible_2_to_2(cipher, target, context):
    target_scale = _require_target_scale(target, "_align_flexible_2_to_2")
    scf = context.scale_at(cipher.cur_limbs)
    q1 = context.rescale_divisor_at(cipher.cur_limbs - 1)
    correction_factor = target_scale / cipher.scaling_factor * q1 / scf
    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    cipher = _rescale_one_level(cipher, context)
    if cipher.cur_limbs > target.cur_limbs:
        cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return _with_target_scale(cipher, target)


def _align_flexible_1_to_1(cipher, target, context):
    if has_target_scale(target):
        pre_rescale_target_sf = target.scaling_factor * context.rescale_divisor_at(target.cur_limbs)
        correction_factor = pre_rescale_target_sf / cipher.scaling_factor / context.scale_at(cipher.cur_limbs)
    else:
        correction_factor = 1.0

    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    if cipher.cur_limbs > target.cur_limbs + 1:
        cipher = _drop_to_limbs(cipher, target.cur_limbs + 1, context)
    cipher = _rescale_one_level(cipher, context)
    return _with_target_scale(cipher, target)


def _align_flexible_2_to_1(cipher, target, context):
    if cipher.cur_limbs == target.cur_limbs + 1:
        cipher = _rescale_one_level(cipher, context)
        return _with_target_scale(cipher, target)

    scf2 = _pre_rescale_target_scale(target, context)
    scf = context.scale_at(cipher.cur_limbs)
    q1 = context.rescale_divisor_at(cipher.cur_limbs - 1)
    correction_factor = scf2 / cipher.scaling_factor * q1 / scf
    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    cipher = _rescale_one_level(cipher, context)
    if cipher.cur_limbs > target.cur_limbs + 1:
        cipher = _drop_to_limbs(cipher, target.cur_limbs + 1, context)
    cipher = _rescale_one_level(cipher, context)
    return _with_target_scale(cipher, target)


def _align_flexible_1_to_2(cipher, target, context):
    target_scale = _require_target_scale(target, "_align_flexible_1_to_2")
    correction_factor = target_scale / cipher.scaling_factor / context.scale_at(cipher.cur_limbs)
    cipher = _multiply_by_scale_correction(cipher, correction_factor, context)
    cipher = _drop_to_limbs(cipher, target.cur_limbs, context)
    return cipher.cipher_like(cipher.cv, scaling_factor=target_scale)
