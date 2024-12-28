from .Ciphertext import Cipher
from . import functional as F

def cipher_check_and_adjust_level(ct1: Cipher, ct2: Cipher, cryptoContext):
    rct1 = Cipher([ct1.cv[0].clone(), ct1.cv[1].clone()], ct1.cur_limbs)
    rct2 = Cipher([ct2.cv[0].clone(), ct2.cv[1].clone()], ct2.cur_limbs)

    if rct1.cur_limbs > rct2.cur_limbs:
        rct1=cipher_level_reduce(rct1, rct1.cur_limbs - rct2.cur_limbs)
    elif rct1.cur_limbs < rct2.cur_limbs:
        rct2=cipher_level_reduce(rct2, rct2.cur_limbs - rct1.cur_limbs)
    return rct1, rct2

def cipher_rescale(ct, cryptoContext):
    res0 = F.cv_rescale(ct.cv[0], cryptoContext, ct.cur_limbs)
    res1 = F.cv_rescale(ct.cv[1], cryptoContext, ct.cur_limbs)
    return Cipher([res0, res1], ct.cur_limbs - 1)


def cipher_mod_reduce(ct, levels, cryptoContext):
    curr_limbs = ct.cur_limbs
    for l in range(levels):
        res0 = F.cv_drop_last_element_and_scale(ct.cv[0], cryptoContext, curr_limbs, l)
        res1 = F.cv_drop_last_element_and_scale(ct.cv[1], cryptoContext, curr_limbs, l)
        curr_limbs -= 1
    return Cipher([res0, res1], curr_limbs)


def cipher_level_reduce(ct, levels):
    return Cipher(ct.cv, ct.cur_limbs - levels)


def cipher_add(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_add(cv0, cv1, cryptoContext.moduliQ_cuda, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs)


def cipher_sub(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_sub(cv0, cv1, cryptoContext.moduliQ_cuda, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs)


def cipher_mul(in0, in1, cryptoContext):
    assert len(in0.cv) == 2 and len(in1.cv) == 2
    assert in0.cur_limbs == in1.cur_limbs
    bx = F.cv_mul(
        in0.cv[0], in1.cv[0], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )
    ax = F.cv_add(
        F.cv_mul(
            in0.cv[0],
            in1.cv[1],
            cryptoContext.moduliQ_cuda,
            cryptoContext.q_mu_cuda,
            in0.cur_limbs,
        ),
        F.cv_mul(
            in0.cv[1],
            in1.cv[0],
            cryptoContext.moduliQ_cuda,
            cryptoContext.q_mu_cuda,
            in0.cur_limbs,
        ),
        cryptoContext.moduliQ_cuda,
        in0.cur_limbs,
    )
    axax = F.cv_mul(
        in0.cv[1], in1.cv[1], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )
    return Cipher([bx, ax, axax], in0.cur_limbs)


def cipher_square(in0, cryptoContext):
    assert len(in0.cv) == 2
    bx = F.cv_mul(
        in0.cv[0], in0.cv[0], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )
    ax = F.cv_mul(
        in0.cv[0],
        in0.cv[1],
        cryptoContext.moduliQ_cuda,
        cryptoContext.q_mu_cuda,
        in0.cur_limbs,
    )
    ax = F.cv_add(ax, ax, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    axax = F.cv_mul(
        in0.cv[1], in0.cv[1], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )

    return Cipher([bx, ax, axax], in0.cur_limbs)


def cipher_add_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ, in0.cur_limbs)
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ_cuda, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs)


def cipher_sub_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ_cuda, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs)


def cipher_mul_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    cv = [
        F.cv_mul_scalar(
            cv0, scalar_mod, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
        )
        for cv0 in in0.cv
    ]
    return Cipher(cv, in0.cur_limbs)


def cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ_cuda, in0.cur_limbs) for cv0 in in0.cv]
    return Cipher(cv, in0.cur_limbs)


def homo_add(in0, in1, cryptoContext):
    if in0.cur_limbs != in1.cur_limbs: #fixme: judgement should be changed to use scaling factor and limbs together after including scalingtechnique flexibleauto
        in0, in1 = cipher_check_and_adjust_level(in0, in1, cryptoContext)

    return cipher_add(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    if in0.cur_limbs != in1.cur_limbs: #fixme: judgement should be changed to use scaling factor and limbs together after including scalingtechnique flexibleauto
        in0, in1 = cipher_check_and_adjust_level(in0, in1, cryptoContext)

    return cipher_sub(in0, in1, cryptoContext)


def homo_mul(in0, in1, cryptoContext):
    if in0.cur_limbs != in1.cur_limbs: #fixme: judgement should be changed to use scaling factor and limbs together after including scalingtechnique flexibleauto
        in0, in1 = cipher_check_and_adjust_level(in0, in1, cryptoContext)

    res = cipher_mul(in0, in1, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx_cuda,
            cryptoContext.swk_ax_cuda,
            cryptoContext,
        ),
        cur_limbs=in0.cur_limbs,
    )
    res.cv = res.cv[:2]
    return cipher_add(res, tmp, cryptoContext)


def homo_square(in0, cryptoContext):
    res = cipher_square(in0, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx_cuda,
            cryptoContext.swk_ax_cuda,
            cryptoContext,
        ),
        cur_limbs=in0.cur_limbs,
    )
    res.cv = res.cv[:2]
    return cipher_add(res, tmp, cryptoContext)


def cpp_round(_float, _len=0):
    i = int(_float)
    if isinstance(_float, float):
        return i if ((_float - i) < 0.5) else i + 1
    else:
        return round(_float, _len)


def homo_add_scalar_double(ct, cnst, cryptoContext):
    tmpr = cpp_round(abs(cnst) * (2 ** cryptoContext.logqi))
    if cnst < 0:
        res = cipher_sub_scalar(ct, tmpr, cryptoContext).cv
    else:
        res = cipher_add_scalar(ct, tmpr, cryptoContext).cv

    return Cipher(res, ct.cur_limbs)


def homo_mul_scalar_int(in0, scalar, cryptoContext):
    res = cipher_mul_scalar(in0, scalar, cryptoContext)
    if scalar < 0:
        res = cipher_neg(res, cryptoContext)
    return Cipher(res.cv, in0.cur_limbs)


def homo_mul_scalar_double(in0, scalar, cryptoContext):
    tmpr = cpp_round(abs(scalar) * (2 ** cryptoContext.logqi))
    res = cipher_mul_scalar(in0, tmpr, cryptoContext)
    if scalar < 0:
        res = cipher_neg(res, cryptoContext)
    return Cipher(res.cv, in0.cur_limbs)

def homo_rotate(cipher, auto_index, ctx):
    cur_limbs = cipher.cur_limbs
    swk = ctx.left_rot_key_map[str(auto_index)]
    res = F.cv_keyswitch(cipher.cv[1], cur_limbs, swk[0], swk[1], ctx)
    bxrot = F.cv_add(cipher.cv[0], res[0], ctx.moduliQ_cuda, cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    cv0 = F.cv_automorphism_transform(bxrot, cur_limbs, auto_index, ctx)
    cv1 = F.cv_automorphism_transform(res[1], cur_limbs, auto_index, ctx)

    return Cipher([cv0, cv1], cur_limbs)

def homo_conjugate(cipher, auto_index, ctx):
    return homo_rotate(cipher, auto_index, ctx)