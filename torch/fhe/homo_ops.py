import torch
import numpy as np
from .Ciphertext import Ciphertext, Cipher
from .context import Context
from . import functional as F
from .data.bsConst import *


def timeit(func):
    def wrapper(*args, **kw):
        import time

        start = time.time()
        print(f"Start {func.__name__}")
        res = func(*args, **kw)
        end = time.time()
        print(f"{func.__name__} time: {end - start}")
        return res

    return wrapper


def ct_convert(func):
    def wrapper(*args, **kw):
        args_list = list(args)
        for i in range(len(args_list)):
            if isinstance(args_list[i], Ciphertext):
                dim = args_list[i].cv.shape[0]
                cv = [torch.from_numpy(args_list[i].cv[j]).cuda() for j in range(dim)]
                cv[0], cv[1] = cv[1], cv[0]
                cipher = Cipher(cv, args_list[i].curr_limbs)
                args_list[i] = cipher
            if isinstance(args_list[i], Context):
                cryptoContext = args_list[i]
                args_list[i].moduliQ = torch.from_numpy(args_list[i].moduliQ).cuda()
        new_args = tuple(args_list)
        res = func(*new_args, **kw)
        cryptoContext.moduliQ = cryptoContext.moduliQ.cpu().numpy()
        cv = res.cv
        cv[0], cv[1] = cv[1], cv[0]
        return Ciphertext(np.array(cv), res.cur_limbs)

    return wrapper


# @ct_convert
def cipher_add(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs)


# @ct_convert
def cipher_sub(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs)


# @ct_convert
def cipher_mul(in0, in1, cryptoContext):
    assert len(in0.cv) == 2 and len(in1.cv) == 2
    assert in0.cur_limbs == in1.cur_limbs
    bx = F.cv_mul(
        in0.cv[0], in1.cv[0], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs
    )
    ax = F.cv_add(
        F.cv_mul(
            in0.cv[0],
            in1.cv[1],
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        ),
        F.cv_mul(
            in0.cv[1],
            in1.cv[0],
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        ),
        cryptoContext.moduliQ,
        in0.cur_limbs,
    )
    axax = F.cv_mul(
        in0.cv[1], in1.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs
    )
    return Cipher([bx, ax, axax], in0.cur_limbs)


# @ct_convert
def cipher_square(in0, cryptoContext):
    assert len(in0.cv) == 2
    bx = F.cv_mul(
        in0.cv[0], in0.cv[0], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs
    )
    ax = F.cv_mul(
        in0.cv[0],
        in0.cv[1],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        in0.cur_limbs,
    )
    ax = F.cv_add(ax, ax, cryptoContext.moduliQ, in0.cur_limbs)
    axax = F.cv_mul(
        in0.cv[1], in0.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs
    )

    return Cipher([bx, ax, axax], in0.cur_limbs)


# @ct_convert
def cipher_add_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ, in0.cur_limbs)
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs)


# @ct_convert
def cipher_sub_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ, in0.cur_limbs)
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs)


# @ct_convert
def cipher_mul_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ, in0.cur_limbs)
    cv = [
        F.cv_mul_scalar(
            cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs
        )
        for cv0 in in0.cv
    ]
    return Cipher(cv, in0.cur_limbs)


# @ct_convert
def cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return Cipher(cv, in0.cur_limbs)


# @ct_convert
def rescale_ct(ct, cryptoContext):
    res0 = F.cv_rescale(ct.cv[0], cryptoContext, ct.cur_limbs)
    res1 = F.cv_rescale(ct.cv[1], cryptoContext, ct.cur_limbs)
    return Cipher([res0, res1], ct.cur_limbs - 1)


# @ct_convert
def ModReduce_ct(ct, levels, cryptoContext):
    curr_limbs = ct.cur_limbs
    for l in range(levels):
        res0 = F.cv_drop_last_element_and_scale(ct.cv[0], cryptoContext, curr_limbs, l)
        res1 = F.cv_drop_last_element_and_scale(ct.cv[1], cryptoContext, curr_limbs, l)
        curr_limbs -= 1
    return Cipher([res0, res1], curr_limbs)


def homo_add(in0, in1, cryptoContext):
    return cipher_add(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    return cipher_sub(in0, in1, cryptoContext)


def homo_mul(in0, in1, cryptoContext):
    res = cipher_mul(in0, in1, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(res.cv[2], res.cur_limbs, cryptoContext), cur_limbs=in0.cur_limbs
    )
    res.cv = res.cv[:2]
    return cipher_add(res, tmp, cryptoContext)


def homo_square(in0, cryptoContext):
    res = cipher_square(in0, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(res.cv[2], res.cur_limbs, cryptoContext), cur_limbs=in0.cur_limbs
    )
    res.cv = res.cv[:2]
    return cipher_add(res, tmp, cryptoContext)


def LevelReduce_ct(ct, levels):
    return Cipher(ct.cv, ct.cur_limbs - levels)


def homo_add_scalar_double(ct, cnst, cryptoContext):
    tmpr = int(abs(cnst) * (2**cryptoContext.logqi))
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
    tmpr = abs(scalar) * (2**cryptoContext.logqi)
    res = cipher_mul_scalar(in0, tmpr, cryptoContext)
    if scalar < 0:
        res = cipher_neg(res, cryptoContext)
    return Cipher(res.cv, in0.cur_limbs)
