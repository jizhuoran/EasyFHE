import torch
import numpy as np
from .Ciphertext import Ciphertext, Cipher
from .context import Context
from . import functional as F
from . import arithmetic
from . import KeySwitch
from .data.bsConst import *

Tensor = torch.Tensor


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
                # if isinstance(cryptoContext.moduliQ, Tensor):
                #     pass
                # else:
                args_list[i].moduliQ = torch.from_numpy(args_list[i].moduliQ).cuda()
        new_args = tuple(args_list)
        res = func(*new_args, **kw)
        # if isinstance(cryptoContext.moduliQ, Tensor):
        cryptoContext.moduliQ = cryptoContext.moduliQ.cpu().numpy()
        cv = res.cv
        cv[0], cv[1] = cv[1], cv[0]
        # if isinstance(res, Ciphertext):
        #     return Ciphertext(cv, res.curr_limbs)
        return Ciphertext(np.array(cv), res.cur_limbs)

    return wrapper


@ct_convert
def cipher_add(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs)


@ct_convert
def cipher_sub(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs)


@ct_convert
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


@ct_convert
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


@ct_convert
def cipher_add_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = torch.from_numpy(
        np.array(
            [int(int(scalar) % int(n)) for n in list(cryptoContext.moduliQ.cpu().numpy())],
            dtype=np.uint64,
        )
    ).cuda()
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1].cpu().numpy(),
    ]
    return Cipher(cv, in0.cur_limbs)


@ct_convert
def cipher_sub_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = torch.from_numpy(
        np.array(
            [int(int(scalar) % int(n)) for n in list(cryptoContext.moduliQ.cpu().numpy())],
            dtype=np.uint64,
        )
    ).cuda()
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1].cpu().numpy(),
    ]
    return Cipher(cv, in0.cur_limbs)


@ct_convert
def cipher_mul_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = torch.from_numpy(
        np.array(
            [int(int(scalar) % int(n)) for n in list(cryptoContext.moduliQ.cpu().numpy())],
            dtype=np.uint64,
        )
    ).cuda()
    cv = [
        F.cv_mul_scalar(
            cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs
        )
        for cv0 in in0.cv
    ]
    return Cipher(cv, in0.cur_limbs)


@ct_convert
def cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return Cipher(cv, in0.cur_limbs)


def homo_add(in0, in1, cryptoContext):
    return cipher_add(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    return cipher_sub(in0, in1, cryptoContext)


def homo_mul(in0, in1, cryptoContext):
    res = cipher_mul(in0, in1, cryptoContext)

    tmp = KeySwitch.KeySwitch_core(
        res.cv[2],
        np.array(cryptoContext.mult_swk, dtype=np.uint64),
        cryptoContext.moduliQ,
        cryptoContext.qInvVec,
        cryptoContext.qRootScalePows,
        cryptoContext.qRootScalePowsInv,
        cryptoContext.NScaleInvModq,
        cryptoContext.QHatInvModq,
        cryptoContext.pHatModq,
        cryptoContext.PInvModq,
        cryptoContext.moduliP,
        cryptoContext.pInvVec,
        cryptoContext.pRootScalePows,
        cryptoContext.pRootScalePowsInv,
        cryptoContext.QHatModp,
        cryptoContext.NScaleInvModp,
        cryptoContext.pHatInvModp,
        res.curr_limbs,
        cryptoContext.K,
        cryptoContext.N,
    )

    res.cv = res.cv[:2]
    tmp = Ciphertext(np.array([tmp[0].copy(), tmp[1].copy()]), res.curr_limbs)
    return cipher_add(res, tmp, cryptoContext)


# @ct_convert
def homo_square(in0, cryptoContext):
    res = cipher_square(in0, cryptoContext)
    tmp = KeySwitch.KeySwitch_core(
        res.cv[2],
        np.array(cryptoContext.mult_swk, dtype=np.uint64),
        cryptoContext.moduliQ,
        cryptoContext.qInvVec,
        cryptoContext.qRootScalePows,
        cryptoContext.qRootScalePowsInv,
        cryptoContext.NScaleInvModq,
        cryptoContext.QHatInvModq,
        cryptoContext.pHatModq,
        cryptoContext.PInvModq,
        cryptoContext.moduliP,
        cryptoContext.pInvVec,
        cryptoContext.pRootScalePows,
        cryptoContext.pRootScalePowsInv,
        cryptoContext.QHatModp,
        cryptoContext.NScaleInvModp,
        cryptoContext.pHatInvModp,
        res.curr_limbs,
        cryptoContext.K,
        cryptoContext.N,
    )

    res.cv = res.cv[:2]
    tmp = Ciphertext(np.array([tmp[0].copy(), tmp[1].copy()]), res.curr_limbs)
    return cipher_add(res, tmp, cryptoContext)


def rescale(
    a,
    moduliQ,
    qInvVec,
    qRootScalePows,
    qRootScalePowsInv,
    NScaleInvModq,
    qInvModq,
    curr_limbs,
    N,
):
    res = np.zeros((curr_limbs - 1, N), dtype=np.uint64)

    intt_a_last = arithmetic.iNTT(
        a[curr_limbs - 1],
        N,
        moduliQ[curr_limbs - 1],
        qInvVec[curr_limbs - 1],
        qRootScalePowsInv[curr_limbs - 1],
        NScaleInvModq[curr_limbs - 1],
    )

    for i in range(curr_limbs - 1):
        tmp = arithmetic.vec_mod(intt_a_last, moduliQ[i])
        ntt_tmp = arithmetic.NTT(tmp, N, moduliQ[i], qInvVec[i], qRootScalePows[i])

        res[i] = arithmetic.vec_sub_mod(a[i], ntt_tmp, moduliQ[i])
        res[i] = arithmetic.vec_mul_scalar_mod(
            res[i], qInvModq[curr_limbs - 1][i], moduliQ[i]
        )

    return res


def rescale_ct(ct, cryptoContext):
    assert ct.curr_limbs > 1

    N = cryptoContext.N
    moduliQ = cryptoContext.moduliQ
    qInvVec = cryptoContext.qInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    qInvModq = cryptoContext.qInvModq

    res = np.zeros((2, ct.curr_limbs - 1, N), dtype=np.uint64)
    for k in range(2):
        res[k] = rescale(
            ct.cv[k],
            moduliQ,
            qInvVec,
            qRootScalePows,
            qRootScalePowsInv,
            NScaleInvModq,
            qInvModq,
            ct.curr_limbs,
            N,
        )
    return Ciphertext(res, ct.curr_limbs - 1)


def DropLastElementAndScale(a, cryptoContext, curr_limbs, l):
    # the same as openfhe: DropLastElementAndScale
    res = np.zeros((curr_limbs - 1, cryptoContext.N), dtype=np.uint64)

    intt_a_last = arithmetic.iNTT(
        a[curr_limbs - 1],
        cryptoContext.N,
        cryptoContext.moduliQ[curr_limbs - 1],
        cryptoContext.qInvVec[curr_limbs - 1],
        cryptoContext.qRootScalePowsInv[curr_limbs - 1],
        cryptoContext.NScaleInvModq[curr_limbs - 1],
    )
    for i in range(curr_limbs - 1):
        tmp = arithmetic.vec_switch_modulus(
            intt_a_last, cryptoContext.moduliQ[i], cryptoContext.moduliQ[curr_limbs - 1]
        )
        tmp = arithmetic.vec_mul_scalar_mod(
            tmp,
            cryptoContext.QlQlInvModqlDivqlModq[cryptoContext.L - curr_limbs + l][i],
            cryptoContext.moduliQ[i],
        )
        res[i] = arithmetic.NTT(
            tmp,
            cryptoContext.N,
            cryptoContext.moduliQ[i],
            cryptoContext.qInvVec[i],
            cryptoContext.qRootScalePows[i],
        )

    for i in range(curr_limbs - 1):
        tmp = arithmetic.vec_mul_scalar_mod(
            a[i], cryptoContext.qInvModq[curr_limbs - 1][i], cryptoContext.moduliQ[i]
        )
        res[i] = arithmetic.vec_add_mod(res[i], tmp, cryptoContext.moduliQ[i])

    return res


def ModReduce_ct(ct, levels, cryptoContext):
    assert ct.curr_limbs > 1

    curr_limbs = ct.curr_limbs
    N = cryptoContext.N

    res = np.zeros((2, curr_limbs - 1, N), dtype=np.uint64)
    for l in range(levels):
        for k in range(2):
            res[k] = DropLastElementAndScale(
                ct.cv[k],
                cryptoContext,
                curr_limbs,
                l,
            )
        curr_limbs -= 1
    return Ciphertext(res, curr_limbs)


def LevelReduce_ct(ct, levels):
    return Ciphertext(ct.cv, ct.curr_limbs - levels)

def homo_add_scalar_double(ct, cnst, cryptoContext):
    tmpr = int(abs(cnst) * (2**cryptoContext.logqi))
    MOD = cryptoContext.moduliQ[: ct.curr_limbs]

    if cnst < 0:
        res = cipher_sub_scalar(ct, tmpr, cryptoContext).cv
    else:
        res = cipher_add_scalar(ct, tmpr, cryptoContext).cv

    return Ciphertext(res, ct.curr_limbs)


def homo_mul_scalar_int(in0, scalar, cryptoContext):
    res = cipher_mul_scalar(in0, scalar, cryptoContext)
    if scalar < 0:
        res = cipher_neg(res, cryptoContext)
    return Ciphertext(np.array(res.cv), in0.curr_limbs)


def homo_mul_scalar_double(in0, scalar, cryptoContext):
    tmpr = abs(scalar) * (2**cryptoContext.logqi)
    res = cipher_mul_scalar(in0, tmpr, cryptoContext)
    if scalar < 0:
        res = cipher_neg(res, cryptoContext)
    return Ciphertext(np.array(res.cv), in0.curr_limbs)


def KeySwitch_ct(axax, cryptoContext):
    res = KeySwitch.KeySwitch_core(
        axax,
        cryptoContext.mult_swk,
        cryptoContext.moduliQ,
        cryptoContext.qInvVec,
        cryptoContext.qRootScalePows,
        cryptoContext.qRootScalePowsInv,
        cryptoContext.NScaleInvModq,
        cryptoContext.QHatInvModq,
        cryptoContext.pHatModq,
        cryptoContext.PInvModq,
        cryptoContext.moduliP,
        cryptoContext.pInvVec,
        cryptoContext.pRootScalePows,
        cryptoContext.pRootScalePowsInv,
        cryptoContext.QHatModp,
        cryptoContext.NScaleInvModp,
        cryptoContext.pHatInvModp,
        axax.shape[0],
        cryptoContext.K,
        cryptoContext.N,
    )
    return res


def InnerL1(
    T,
    T2,
    cryptoContext,
    divcs_q,
    s_divcs_q,
    s_divqr_q,
    q_divcs_q,
    q_divqr_q,
    q_s2,
    s_s2,
    s_lg2_divqr_q_last,
    q_lg2_divqr_q_last,
):
    # weightedSum Core computation
    # level = T[4] -1
    tmp = LevelReduce_ct(T[4], 1)
    q_su = homo_mul_scalar_double(tmp, s_divcs_q[5], cryptoContext)
    qs_qu = homo_mul_scalar_double(tmp, s_divqr_q[5], cryptoContext)

    # level = T[4]
    q_qu = homo_mul_scalar_double(T[4], q_divcs_q[5], cryptoContext)
    qq_qu = homo_mul_scalar_double(T[4], q_divqr_q[5], cryptoContext)

    for i in range(4):
        # level = T[i] -1
        tmp1 = LevelReduce_ct(T[i], 1)
        tmp = homo_mul_scalar_double(tmp1, s_divcs_q[i + 1], cryptoContext)
        q_su = homo_add(q_su, tmp, cryptoContext)
        tmp = homo_mul_scalar_double(tmp1, s_divqr_q[i + 1], cryptoContext)
        qs_qu = homo_add(qs_qu, tmp, cryptoContext)

        # level = T[i]
        tmp = homo_mul_scalar_double(T[i], q_divcs_q[i + 1], cryptoContext)
        q_qu = homo_add(q_qu, tmp, cryptoContext)

        tmp = homo_mul_scalar_double(T[i], q_divqr_q[i + 1], cryptoContext)
        # NOTE: set the last element of q_divqr_q zero
        qq_qu = homo_add(qq_qu, tmp, cryptoContext)

    qq_qu = rescale_ct(qq_qu, cryptoContext)
    qs_qu = rescale_ct(qs_qu, cryptoContext)

    tmp = LevelReduce_ct(T[5], 1)
    tmp = homo_mul_scalar_int(tmp, (1 << int(s_lg2_divqr_q_last)), cryptoContext)
    qs_qu = homo_add(qs_qu, tmp, cryptoContext)
    qs_qu = homo_add_scalar_double(qs_qu, s_divqr_q[0], cryptoContext)

    tmp = T[5]
    tmp = homo_mul_scalar_int(tmp, (1 << int(q_lg2_divqr_q_last)), cryptoContext)
    qq_qu = homo_add(qq_qu, tmp, cryptoContext)
    qq_qu = homo_add_scalar_double(qq_qu, q_divqr_q[0], cryptoContext)

    q_su = rescale_ct(q_su, cryptoContext)
    q_su = homo_add_scalar_double(q_su, s_divcs_q[0], cryptoContext)
    tmp = LevelReduce_ct(T2[1], 1)
    q_su = homo_add(q_su, tmp, cryptoContext)

    q_qu = rescale_ct(q_qu, cryptoContext)
    q_qu = homo_add_scalar_double(q_qu, q_divcs_q[0], cryptoContext)
    q_qu = homo_add(q_qu, T2[1], cryptoContext)

    q_qu = homo_mul(q_qu, qq_qu, cryptoContext)
    tmp = LevelReduce_ct(T[4], 1)
    qq_su = homo_mul_scalar_double(tmp, q_s2[5], cryptoContext)

    for i in range(4):
        tmp = LevelReduce_ct(T[i], 1)
        tmp = homo_mul_scalar_double(tmp, q_s2[i + 1], cryptoContext)
        qq_su = homo_add(qq_su, tmp, cryptoContext)

    q_qu = homo_add(q_qu, qq_su, cryptoContext)
    q_qu = rescale_ct(q_qu, cryptoContext)
    tmp = LevelReduce_ct(T[5], 1)
    q_qu = homo_add(q_qu, tmp, cryptoContext)
    q_qu = homo_add_scalar_double(q_qu, q_s2[0], cryptoContext)

    # lazy KS: merge KS in computing `q_su` and `qu`
    tmp = LevelReduce_ct(T[4], 1)
    qu = homo_mul_scalar_double(tmp, divcs_q[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = LevelReduce_ct(T[i], 1)
        tmp = homo_mul_scalar_double(tmp, divcs_q[i + 1], cryptoContext)
        qu = homo_add(qu, tmp, cryptoContext)
    qu = rescale_ct(qu, cryptoContext)
    qu = homo_add_scalar_double(qu, divcs_q[0], cryptoContext)
    qu = homo_add(qu, T2[2], cryptoContext)

    tmp_qu = cipher_mul(qu, q_qu, cryptoContext)
    tmp_q_su = cipher_mul(q_su, qs_qu, cryptoContext)
    tmp = LevelReduce_ct(T[4], 2)
    qs_su = homo_mul_scalar_double(tmp, s_s2[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = LevelReduce_ct(T[i], 2)
        tmp = homo_mul_scalar_double(tmp, s_s2[i + 1], cryptoContext)
        qs_su = homo_add(qs_su, tmp, cryptoContext)

    qu = cipher_add(tmp_qu, tmp_q_su, cryptoContext)
    axax = qu.drop_axax()

    qu = cipher_add(qu, qs_su, cryptoContext)

    summult = KeySwitch_ct(axax, cryptoContext)
    summult = Ciphertext(summult, q_qu.curr_limbs)
    qu = cipher_add(qu, summult, cryptoContext)

    qu = rescale_ct(qu, cryptoContext)
    tmp = LevelReduce_ct(T[5], 2)
    qu = homo_add(qu, tmp, cryptoContext)
    qu = homo_add_scalar_double(qu, s_s2[0], cryptoContext)

    return qu


def EvalChebyshevSeries(x, cryptoContext):
    curr_limbs = x.curr_limbs

    tmp = np.zeros(x.cv.shape, dtype=np.uint64)
    T = [
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    # no linear transformation is needed if a = -1, b = 1, T_1(y) = y
    T[0] = x
    # Computes Chebyshev polynomials up to degree k
    # for y: T_1(y) = y, T_2(y), ... , T_k(y)
    # uses binary tree multiplication
    T[1] = homo_square(T[0], cryptoContext)
    T[1] = homo_add(T[1], T[1], cryptoContext)
    T[1] = rescale_ct(T[1], cryptoContext)
    T[1] = homo_add_scalar_double(T[1], -1.0, cryptoContext)

    T[0] = LevelReduce_ct(T[0], 1)
    T[2] = homo_mul(T[0], T[1], cryptoContext)
    T[2] = homo_add(T[2], T[2], cryptoContext)
    T[2] = rescale_ct(T[2], cryptoContext)
    T[0] = LevelReduce_ct(T[0], 1)
    T[2] = homo_sub(T[2], T[0], cryptoContext)
    T[1] = LevelReduce_ct(T[1], 1)

    T[3] = homo_square(T[1], cryptoContext)
    T[3] = homo_add(T[3], T[3], cryptoContext)
    T[3] = rescale_ct(T[3], cryptoContext)
    T[3] = homo_add_scalar_double(T[3], -1.0, cryptoContext)

    T[0] = LevelReduce_ct(T[0], 1)

    T[4] = homo_mul(T[1], T[2], cryptoContext)
    T[4] = homo_add(T[4], T[4], cryptoContext)
    T[4] = rescale_ct(T[4], cryptoContext)
    T[4] = homo_sub(T[4], T[0], cryptoContext)

    T[5] = homo_square(T[2], cryptoContext)
    T[5] = homo_add(T[5], T[5], cryptoContext)
    T[5] = rescale_ct(T[5], cryptoContext)
    T[5] = homo_add_scalar_double(T[5], -1.0, cryptoContext)

    T[1] = LevelReduce_ct(T[1], 1)
    T[2] = LevelReduce_ct(T[2], 1)

    T2 = [
        None,
        None,
        None,
        None,
    ]

    # Compute the Chebyshev polynomials T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    T2[1] = homo_square(T[5], cryptoContext)
    T[5] = LevelReduce_ct(T[5], 1)
    T2[1] = homo_add(T2[1], T2[1], cryptoContext)
    T2[1] = rescale_ct(T2[1], cryptoContext)
    T2[1] = homo_add_scalar_double(T2[1], -1.0, cryptoContext)

    # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
    tmpct2 = T2[1]
    T2km1 = T[5]
    tmpct2 = homo_add(tmpct2, tmpct2, cryptoContext)
    tmpct2 = homo_add_scalar_double(tmpct2, -1.0, cryptoContext)
    T2km1 = homo_mul(T2km1, tmpct2, cryptoContext)
    T2km1 = rescale_ct(T2km1, cryptoContext)

    for i in range(2, 4):
        square = homo_square(T2[i - 1], cryptoContext)
        T2[i] = homo_add(square, square, cryptoContext)
        T2[i] = rescale_ct(T2[i], cryptoContext)
        T2[i] = homo_add_scalar_double(T2[i], -1.0, cryptoContext)

        # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        tmpct2 = T2[i]
        T2km1 = homo_mul(T2km1, tmpct2, cryptoContext)
        T2km1 = homo_add(T2km1, T2km1, cryptoContext)
        T2km1 = rescale_ct(T2km1, cryptoContext)
        tmpct2 = T[5]
        tmpct2 = LevelReduce_ct(tmpct2, i)
        T2km1 = homo_sub(T2km1, tmpct2, cryptoContext)

    qu = InnerL1(
        T,
        T2,
        cryptoContext,
        q_divcs_q,
        qs_divcs_q,
        qs_divqr_q,
        qq_divcs_q,
        qq_divqr_q,
        qq_s2,
        qs_s2,
        qs_lg2_divqr_q_last,
        qq_lg2_divqr_q_last,
    )
    su = InnerL1(
        T,
        T2,
        cryptoContext,
        s_divcs_q,
        ss_divcs_q,
        ss_divqr_q,
        sq_divcs_q,
        sq_divqr_q,
        sq_s2,
        ss_s2,
        ss_lg2_divqr_q_last,
        sq_lg2_divqr_q_last,
    )

    result = homo_mul_scalar_double(T[0], first_divcs_q[1], cryptoContext)
    for i in range(1, 5):
        tmp = homo_mul_scalar_double(T[i], first_divcs_q[i + 1], cryptoContext)
        result = homo_add(result, tmp, cryptoContext)

    result = LevelReduce_ct(result, 2)
    result = rescale_ct(result, cryptoContext)

    result = homo_add_scalar_double(result, first_divcs_q[0], cryptoContext)
    result = homo_add(result, T2[3], cryptoContext)

    result = homo_mul(result, qu, cryptoContext)
    result = rescale_ct(result, cryptoContext)
    su = LevelReduce_ct(su, 1)
    result = homo_add(result, su, cryptoContext)
    result = homo_sub(result, T2km1, cryptoContext)

    return result


def DoubleAngleIteration(in0, cryptoContext):
    r = int(R)
    scalar = [
        -0.94418452709144784,
        -0.89148442119890103,
        -0.79474447324033948,
        -0.63161877774606467,
        -0.3989422804014327,
        -0.15915494309189535,
    ]

    for j in range(1, r + 1):
        in0 = homo_square(in0, cryptoContext)
        in0 = ModReduce_ct(in0, 1, cryptoContext)
        in0 = homo_add(in0, in0, cryptoContext)
        # scalar = np.float64(np.float64(-1.0) / np.float64(math.pow((2.0 * M_PI), np.float64(math.pow(2.0, j - r)))))
        in0 = homo_add_scalar_double(in0, scalar[j - 1], cryptoContext)

    return in0
