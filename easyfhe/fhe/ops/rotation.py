from . import kernels as F
from ..runtime.instrumentation import run_instrumented_op
from . import keyswitch


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


def eval_fast_rotate(digits, cipher, index, need_KS_add, need_moddown, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "eval_fast_rotate",
        _eval_fast_rotate,
        digits,
        cipher,
        index,
        need_KS_add,
        need_moddown,
        cryptoContext,
    )


def _eval_fast_rotate(digits, cipher, index, need_KS_add, need_moddown, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    result = keyswitch._mult_rot_key_and_sum_ext(digits, index, cryptoContext)

    if need_KS_add:
        if need_moddown and cipher.is_ext:
            raise ValueError("eval_fast_rotate: need_moddown=True is incompatible with ext cipher")
        if cipher.is_ext == True:
            result.cv[0] = F.cv_add(
                result.cv[0],
                cipher.cv[0],
                cryptoContext.QplusP_map[cipher.cur_limbs],
                cipher.cur_limbs + cryptoContext.K,
            )
        else:
            if need_moddown:
                result = keyswitch.moddown_from_ext(result, cryptoContext)
                cipher_cv0 = cipher.cv[0]
            else:
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

    result = _cipher_automorphism(result, index, cryptoContext)
    return result


def homo_conjugate(in0, cryptoContext):
    return run_instrumented_op(cryptoContext, "homo_conjugate", _homo_conjugate, in0, cryptoContext)


def _homo_conjugate(in0, cryptoContext):
    return homo_rotate(in0, 2 * cryptoContext.N - 1, cryptoContext)
