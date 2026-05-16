import easyfhe as torch

from . import kernels as F
from ..runtime.instrumentation import run_instrumented_op


def _cipher_automorphism(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    limbs = in0.cur_limbs if in0.is_ext == False else in0.cur_limbs + cryptoContext.K
    cv = [F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext) for cv in in0.cv]
    return in0.cipher_like(cv)


def cipher_automorphism(in0, index, cryptoContext):
    return run_instrumented_op(cryptoContext, "cipher_automorphism", _cipher_automorphism_public, in0, index, cryptoContext)


def _cipher_automorphism_public(in0, index, cryptoContext):
    return _cipher_automorphism(in0, index, cryptoContext)


def homo_rotate(in0, index, cryptoContext):
    return run_instrumented_op(cryptoContext, "homo_rotate", _homo_rotate, in0, index, cryptoContext)


def _homo_rotate(in0, index, cryptoContext):
    if index == 0:
        return in0.deep_copy()
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.get_rotation_key(norm_index)
    special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(index, cryptoContext.L)
    res = in0.cipher_like(F.cv_keyswitch(in0.cv[1], in0.cur_limbs, special_mod_start, swk[0], swk[1], cryptoContext))
    res.cv[0] = F.cv_add(in0.cv[0], res.cv[0], cryptoContext.moduliQ, in0.cur_limbs)
    res = _cipher_automorphism(res, index, cryptoContext)
    return res


def _fast_rotation_key_product(digits, index, cryptoContext):
    if not digits.is_ext:
        raise ValueError("fast rotation key product: expected ext digits")
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.get_rotation_key(norm_index)
    special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(index, cryptoContext.L)
    sum_mult = F.cv_innerproduct(
        digits.cv[0].reshape(-1),
        curr_limbs=digits.cur_limbs,
        special_mod_start=special_mod_start,
        context=cryptoContext,
        swk_bx=swk[0],
        swk_ax=swk[1],
    )
    return digits.cipher_like(sum_mult, is_ext=True)


def _key_switch_P_ext(cipher, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "key_switch_P_ext",
        _key_switch_P_ext_impl,
        cipher,
        cryptoContext,
    )


def _key_switch_P_ext_impl(cipher, cryptoContext):
    if cipher.is_ext:
        raise ValueError("key_switch_P_ext: expected non-ext cipher")
    cv = [
        torch.cat((
            F.cv_mul_scalar(cv, cryptoContext.PModq, cryptoContext.moduliQ, cryptoContext.q_mu, cipher.cur_limbs),
            torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device=cryptoContext.device).reshape(-1, cryptoContext.N)
        ), dim=0)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=True)


def _modup_to_ext(cipher, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "modup_to_ext",
        _modup_to_ext_impl,
        cipher,
        cryptoContext,
    )


def _modup_to_ext_impl(cipher, cryptoContext):
    if cipher.is_ext:
        raise ValueError("modup_to_ext: expected non-ext cipher")
    if len(cipher.cv) < 2:
        raise ValueError(f"modup_to_ext: expected at least two components, got {len(cipher.cv)}")
    cv = [F.cv_modup(cipher.cv[1], cipher.cur_limbs, cryptoContext)]
    return cipher.cipher_like(cv, is_ext=True)


def _moddown_from_ext(cipher, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "moddown_from_ext",
        _moddown_from_ext_impl,
        cipher,
        cryptoContext,
    )


def moddown_from_ext(cipher, cryptoContext):
    return _moddown_from_ext(cipher, cryptoContext)


def _moddown_from_ext_impl(cipher, cryptoContext):
    if not cipher.is_ext:
        raise ValueError("moddown_from_ext: expected ext cipher")
    cv = [
        F.cv_moddown(cv, cipher.cur_limbs, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=False)


def fast_rotate(cipher, offsets, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "fast_rotate",
        _fast_rotate,
        cipher,
        offsets,
        cryptoContext,
    )


def fast_rotate_ext(cipher, offsets, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "fast_rotate_ext",
        _fast_rotate_ext_public,
        cipher,
        offsets,
        cryptoContext,
    )


def _fast_rotate_ext_public(cipher, offsets, cryptoContext):
    if cipher is None or cipher.is_ext:
        raise ValueError("fast_rotate_ext: expected a non-ext cipher")
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        return []
    digits = _modup_to_ext(cipher, cryptoContext)
    return [
        _fast_rotate_ext(digits, cipher, index, cryptoContext)
        for index in offsets
    ]


def _fast_rotate(cipher, offsets, cryptoContext):
    if cipher is None or cipher.is_ext:
        raise ValueError("fast_rotate: expected a non-ext cipher")
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        return []
    digits = _modup_to_ext(cipher, cryptoContext)
    return [
        _fast_rotate_from_digits(digits, cipher, index, cryptoContext)
        for index in offsets
    ]


def _fast_rotate_from_digits(digits, cipher, index, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    result = _fast_rotation_key_product(digits, index, cryptoContext)
    result = _moddown_from_ext(result, cryptoContext)
    result.cv[0] = F.cv_add(
        result.cv[0],
        cipher.cv[0],
        cryptoContext.moduliQ,
        cipher.cur_limbs,
        inplace=True,
    )
    return _cipher_automorphism(result, index, cryptoContext)


def _fast_rotate_key_contribution_ext(digits, index, cryptoContext):
    if index == 0:
        raise ValueError("fast rotate key contribution: rotation index cannot be 0")
    result = _fast_rotation_key_product(digits, index, cryptoContext)
    return _cipher_automorphism(result, index, cryptoContext)


def _fast_rotate_ext(digits, cipher, index, cryptoContext):
    if cipher is None or cipher.is_ext:
        raise ValueError("fast_rotate_ext: expected a non-ext cipher")
    if index == 0:
        return _key_switch_P_ext(cipher, cryptoContext)

    result = _fast_rotation_key_product(digits, index, cryptoContext)
    cipher_cv0 = F.cv_mul_scalar(
        cipher.cv[0],
        cryptoContext.PModq,
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        cipher.cur_limbs,
    )
    result.cv[0] = F.cv_add(
        result.cv[0],
        cipher_cv0,
        cryptoContext.moduliQ,
        cipher.cur_limbs,
        inplace=True,
    )
    return _cipher_automorphism(result, index, cryptoContext)


def double_hoist_rotate_sum(inner_exts, offsets, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "double_hoist_rotate_sum",
        _double_hoist_rotate_sum,
        inner_exts,
        offsets,
        cryptoContext,
    )


def _double_hoist_rotate_sum(inner_exts, offsets, cryptoContext):
    from .arithmetic import homo_add
    from .slots import extract_cv

    inner_exts = tuple(inner_exts)
    offsets = tuple(int(offset) for offset in offsets)
    if len(inner_exts) != len(offsets):
        raise ValueError(
            "double_hoist_rotate_sum: inner_exts and offsets must have the same length, "
            f"got {len(inner_exts)} and {len(offsets)}"
        )
    if not inner_exts:
        raise ValueError("double_hoist_rotate_sum: expected at least one inner ext cipher")

    first_acc = None
    outer_ext = None

    for inner_ext, offset in zip(inner_exts, offsets):
        if not inner_ext.is_ext:
            raise ValueError("double_hoist_rotate_sum: expected ext inner ciphers")
        if len(inner_ext.cv) < 2:
            raise ValueError("double_hoist_rotate_sum: expected two-component inner ciphers")

        if offset == 0:
            inner_ext_cv0 = extract_cv(inner_ext, 0, cryptoContext)
            first = _moddown_from_ext(inner_ext_cv0, cryptoContext)
            c1_ext = extract_cv(inner_ext, 1, cryptoContext, append_zeros=True)
        else:
            inner = _moddown_from_ext(inner_ext, cryptoContext)
            inner_cv0 = extract_cv(inner, 0, cryptoContext)
            first = _cipher_automorphism(inner_cv0, offset, cryptoContext)

            inner_digits = _modup_to_ext(inner, cryptoContext)
            c1_ext = _fast_rotate_key_contribution_ext(
                inner_digits,
                offset,
                cryptoContext,
            )

        first_acc = first if first_acc is None else homo_add(first_acc, first, cryptoContext)
        outer_ext = c1_ext if outer_ext is None else homo_add(outer_ext, c1_ext, cryptoContext)

    outer = _moddown_from_ext(outer_ext, cryptoContext)
    first_full_cv = extract_cv(first_acc, 0, cryptoContext, append_zeros=True)
    return homo_add(outer, first_full_cv, cryptoContext)
