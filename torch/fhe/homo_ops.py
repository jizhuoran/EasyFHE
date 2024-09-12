import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import functional as F
from . import arithmetic
from . import KeySwitch
from .data.bsConst import *

Tensor = torch.Tensor

def polys_add_mod(a, b, MOD):
    assert min(a.shape[0], b.shape[0]) >= len(MOD)
    curr_limbs = len(MOD)
    N = a.shape[1]
    c_ = torch.zeros((curr_limbs,N), dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a[:curr_limbs]).cuda()
    b_ = torch.from_numpy(b[:curr_limbs]).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_add_mod(a_[i], b_[i], val)
    c = c_.cpu().numpy()
    return c

def polys_sub_mod(a, b, MOD):
    assert min(a.shape[0], b.shape[0]) >= len(MOD)
    curr_limbs = len(MOD)
    N = a.shape[1]
    c_ = torch.zeros((curr_limbs,N), dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a[:curr_limbs]).cuda()
    b_ = torch.from_numpy(b[:curr_limbs]).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_sub_mod(a_[i], b_[i], val)
    c = c_.cpu().numpy()
    return c

def polys_mul_mod(a, b, MOD, mu):
    assert min(a.shape[0], b.shape[0]) >= len(MOD)
    curr_limbs = len(MOD)
    N = a.shape[1]
    c_ = torch.zeros((curr_limbs,N), dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a[:curr_limbs]).cuda()
    b_ = torch.from_numpy(b[:curr_limbs]).cuda()
    mu_ = torch.from_numpy(mu).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_mul_mod(a_[i], b_[i], val, mu_[i])
    c = c_.cpu().numpy()
    return c


def polys_add_cnst_mod(a, scalar, MOD):
    assert a.shape[0] >= len(MOD)
    curr_limbs = len(MOD)
    N = a.shape[1]
    c = np.zeros((curr_limbs, N), dtype=np.uint64)
    for i, val in enumerate(MOD):
        c[i] = arithmetic.vec_add_scalar_mod(a[i], scalar, val)
    return c


def polys_sub_cnst_mod(a, scalar, MOD):
    assert a.shape[0] >= len(MOD)
    curr_limbs = len(MOD)
    N = a.shape[1]
    c = np.zeros((curr_limbs, N), dtype=np.uint64)
    for i, val in enumerate(MOD):
        c[i] = arithmetic.vec_sub_scalar_mod(a[i], scalar, val)
    return c


def polys_mul_const_mod(a, cnst, MOD):
    assert a.shape[0] >= len(MOD)
    curr_limbs = len(MOD)
    N = a.shape[1]
    c = np.zeros((curr_limbs, N), dtype=np.uint64)
    for i, val in enumerate(MOD):
        c[i] = arithmetic.vec_mul_scalar_mod(a[i], cnst, val)
    return c

def polys_neg_mod(a, MOD):
    assert a.shape[0] >= len(MOD)
    curr_limbs = len(MOD)
    N = a.shape[1]
    c = np.zeros((curr_limbs, N), dtype=np.uint64)
    for i, val in enumerate(MOD):
        c[i] = arithmetic.vec_neg_mod(a[i], val)
    return c


def homo_add(in0, in1, cryptoContext):
    dim = in0.cv.shape[0]
    curr_limbs = min(in0.curr_limbs, in1.curr_limbs)
    N = in0.cv.shape[2]
    MOD = cryptoContext.moduliQ[:curr_limbs]
    res = np.zeros((dim, curr_limbs, N), dtype=np.uint64)
    res[0] = polys_add_mod(in0.cv[0], in1.cv[0], MOD)  # res.ax
    res[1] = polys_add_mod(in0.cv[1], in1.cv[1], MOD)  # res.bx

    return Ciphertext(res, curr_limbs)

def homo_sub(in0, in1, cryptoContext):
    dim = in0.cv.shape[0]
    curr_limbs = min(in0.curr_limbs, in1.curr_limbs)
    N = in0.cv.shape[2]
    MOD = cryptoContext.moduliQ[:curr_limbs]
    res = np.zeros((dim, curr_limbs, N), dtype=np.uint64)
    res[0] = polys_sub_mod(in0.cv[0], in1.cv[0], MOD)  # res.ax
    res[1] = polys_sub_mod(in0.cv[1], in1.cv[1], MOD)  # res.bx

    return Ciphertext(res, curr_limbs)

# for dim(ct) = 2 only
def homo_mult_core(in0, in1, MOD, MU):
    dim = 3
    curr_limbs = len(MOD)
    N = in0.cv.shape[2]
    res = np.zeros((dim, curr_limbs, N), dtype=np.uint64)

    res[0] = polys_add_mod(in0.cv[0], in0.cv[1], MOD)  # res.ax
    axbx2 = polys_add_mod(in1.cv[0], in1.cv[1], MOD)
    res[0] = polys_mul_mod(res[0], axbx2, MOD, MU)
    res[2] = polys_mul_mod(in0.cv[0], in1.cv[0], MOD, MU)  # axax, use in the next KS step
    res[1] = polys_mul_mod(in0.cv[1], in1.cv[1], MOD, MU)  # res.bx
    tmp = polys_add_mod(res[1], res[2], MOD)
    res[0] = polys_sub_mod(res[0], tmp, MOD)

    return res


def homo_mult(in0, in1,
              cryptoContext):
    curr_limbs = min(in0.curr_limbs, in1.curr_limbs)
    N = cryptoContext.N
    K = cryptoContext.K
    swk = np.array(cryptoContext.mult_swk, dtype=np.uint64)

    moduliQ = cryptoContext.moduliQ
    moduliP = cryptoContext.moduliP
    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq
    q_mu = cryptoContext.q_mu

    res = homo_mult_core(in0, in1, moduliQ[:curr_limbs], q_mu[:curr_limbs])
    tmp = KeySwitch.KeySwitch_core(
        res[2], swk,
        moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq, pHatModq, PInvModq,
        moduliP, pInvVec, pRootScalePows, pRootScalePowsInv, QHatModp, NScaleInvModp, pHatInvModp,
        curr_limbs, K, N)

    res = res[:2]
    MOD = moduliQ[:curr_limbs]
    res[0] = polys_add_mod(res[0], tmp[0], MOD)  # res.ax
    res[1] = polys_add_mod(res[1], tmp[1], MOD)  # res.bx

    return Ciphertext(res, curr_limbs)

def homo_square_core(in0, MOD, MU):
    dim = 3
    curr_limbs = len(MOD)
    N = in0.cv.shape[2]
    res = np.zeros((dim, curr_limbs, N), dtype=np.uint64)

    res[0] = polys_add_mod(in0.cv[0], in0.cv[1], MOD)  # res.ax
    res[0] = polys_mul_mod(res[0], res[0], MOD, MU)
    res[1] = polys_mul_mod(in0.cv[1], in0.cv[1], MOD, MU)  # res.bx
    res[2] = polys_mul_mod(in0.cv[0], in0.cv[0], MOD, MU)  # axax, use in the next KS step
    tmp = polys_add_mod(res[1], res[2], MOD)
    res[0] = polys_sub_mod(res[0], tmp, MOD)

    return res


def homo_square(in0, cryptoContext):

    curr_limbs = in0.curr_limbs
    N = cryptoContext.N
    K = cryptoContext.K
    swk = np.array(cryptoContext.mult_swk, dtype=np.uint64)

    moduliQ = cryptoContext.moduliQ
    moduliP = cryptoContext.moduliP
    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq
    q_mu = cryptoContext.q_mu

    res = homo_square_core(in0, moduliQ[:curr_limbs], q_mu[:curr_limbs])
    tmp = KeySwitch.KeySwitch_core(
        res[2], swk,
        moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq, pHatModq, PInvModq,
        moduliP, pInvVec, pRootScalePows, pRootScalePowsInv, QHatModp, NScaleInvModp, pHatInvModp,
        curr_limbs, K, N)

    res = res[:2]
    MOD = moduliQ[:curr_limbs]
    res[0] = polys_add_mod(res[0], tmp[0], MOD)  # res.ax
    res[1] = polys_add_mod(res[1], tmp[1], MOD)  # res.bx

    return Ciphertext(res, curr_limbs)

def rescale(a,
            moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, qInvModq,
            curr_limbs, N):
    res = np.zeros((curr_limbs - 1, N), dtype=np.uint64)

    intt_a_last = arithmetic.iNTT(a[curr_limbs - 1], N,
                                  moduliQ[curr_limbs - 1], qInvVec[curr_limbs - 1], qRootScalePowsInv[curr_limbs - 1],
                                  NScaleInvModq[curr_limbs - 1])

    for i in range(curr_limbs - 1):
        tmp = arithmetic.vec_mod(intt_a_last, moduliQ[i])
        ntt_tmp = arithmetic.NTT(tmp, N, moduliQ[i], qInvVec[i], qRootScalePows[i])

        res[i] = arithmetic.vec_sub_mod(a[i], ntt_tmp, moduliQ[i])
        res[i] = arithmetic.vec_mul_scalar_mod(res[i], qInvModq[curr_limbs - 1][i], moduliQ[i])

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
        res[k] = rescale(ct.cv[k], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, qInvModq,
                         ct.curr_limbs, N)
    return Ciphertext(res, ct.curr_limbs-1)

def DropLastElementAndScale(
        a,
        moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
        QlQlInvModqlDivqlModq, qInvModq,
        curr_limbs, N):
    # the same as openfhe: DropLastElementAndScale
    res = np.zeros((curr_limbs - 1, N), dtype=np.uint64)

    intt_a_last = arithmetic.iNTT(a[curr_limbs - 1], N,
                                  moduliQ[curr_limbs - 1], qInvVec[curr_limbs - 1], qRootScalePowsInv[curr_limbs - 1],
                                  NScaleInvModq[curr_limbs - 1])
    for i in range(curr_limbs - 1):
        tmp = arithmetic.vec_switch_modulus(intt_a_last, moduliQ[i], moduliQ[curr_limbs - 1])
        tmp = arithmetic.vec_mul_scalar_mod(tmp, QlQlInvModqlDivqlModq[i], moduliQ[i])
        res[i] = arithmetic.NTT(tmp, N, moduliQ[i], qInvVec[i], qRootScalePows[i])

    for i in range(curr_limbs - 1):
        tmp = arithmetic.vec_mul_scalar_mod(a[i], qInvModq[curr_limbs - 1][i], moduliQ[i])
        res[i] = arithmetic.vec_add_mod(res[i], tmp, moduliQ[i])

    return res

def ModReduce_ct(ct, levels, cryptoContext):
    assert ct.curr_limbs > 1

    L = cryptoContext.L
    curr_limbs = ct.curr_limbs
    diffQl = L - curr_limbs
    N = cryptoContext.N
    moduliQ = cryptoContext.moduliQ
    qInvVec = cryptoContext.qInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    qInvModq = cryptoContext.qInvModq
    QlQlInvModqlDivqlModq = cryptoContext.QlQlInvModqlDivqlModq

    res = np.zeros((2, curr_limbs - 1, N), dtype=np.uint64)
    for l in range(levels):
        for k in range(2):
            res[k] = DropLastElementAndScale(
                ct.cv[k], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                QlQlInvModqlDivqlModq[diffQl + l], qInvModq,
                curr_limbs, N)
        curr_limbs-=1
    return Ciphertext(res, curr_limbs)

def LevelReduce_ct(ct, levels):
    assert ct.curr_limbs-levels > 0
    res_limbs = ct.curr_limbs-levels
    N = ct.cv.shape[2]
    res_cv = np.zeros((2, res_limbs, N), dtype=np.uint64)
    res_cv = ct.cv[:, :ct.curr_limbs-levels, :]
    return Ciphertext(res_cv, res_limbs)


def add_cnstDouble_ct(ct, cnst, cryptoContext):
    tmpr = int(abs(cnst) * (2**cryptoContext.logqi))
    MOD = cryptoContext.moduliQ[:ct.curr_limbs]
    res = np.zeros(ct.cv.shape, dtype=np.uint64)

    if cnst < 0:
        res[0] = ct.cv[0]
        res[1] = polys_sub_cnst_mod(ct.cv[1], tmpr, MOD)
    else:
        res[0] = ct.cv[0]
        res[1] = polys_add_cnst_mod(ct.cv[1], tmpr, MOD)

    return Ciphertext(res, ct.curr_limbs)

def mult_cnstDouble_ct(ct, cnst, cryptoContext):
    tmpr = abs(cnst)*(2**cryptoContext.logqi)
    MOD = cryptoContext.moduliQ[:ct.curr_limbs]
    res = np.zeros(ct.cv.shape, dtype=np.uint64)
    res[0] = polys_mul_const_mod(ct.cv[0], tmpr, MOD)
    res[1] = polys_mul_const_mod(ct.cv[1], tmpr, MOD)

    if cnst<0:
        res[0] = polys_neg_mod(res[0], MOD)
        res[1] = polys_neg_mod(res[1], MOD)

    return Ciphertext(res, ct.curr_limbs)

def mult_cnstInt_ct(ct, cnst, cryptoContext):

    MOD = cryptoContext.moduliQ[:ct.curr_limbs]
    res = np.zeros(ct.cv.shape, dtype=np.uint64)
    res[0] = polys_mul_const_mod(ct.cv[0], cnst, MOD)
    res[1] = polys_mul_const_mod(ct.cv[1], cnst, MOD)

    if cnst<0:
        res[0] = polys_neg_mod(res[0], MOD)
        res[1] = polys_neg_mod(res[1], MOD)

    return Ciphertext(res, ct.curr_limbs)


def KeySwitch_ct(axax, cryptoContext):

    K = cryptoContext.K
    N = cryptoContext.N
    curr_limbs = axax.shape[0]
    mult_swk = cryptoContext.mult_swk
    moduliQ = cryptoContext.moduliQ
    moduliP = cryptoContext.moduliP
    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq

    res = KeySwitch.KeySwitch_core(axax, mult_swk, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                   QHatInvModq, pHatModq, PInvModq, moduliP, pInvVec, pRootScalePows, pRootScalePowsInv,
                                   QHatModp, NScaleInvModp, pHatInvModp, curr_limbs, K, N)
    return res


def InnerL1Q(T, T2, cryptoContext):
    # weightedSum Core computation
    # level = T[4] -1
    tmp = LevelReduce_ct(T[4], 1)
    q_su = mult_cnstDouble_ct(tmp, qs_divcs_q[5], cryptoContext)
    qs_qu = mult_cnstDouble_ct(tmp, qs_divqr_q[5], cryptoContext)

    # level = T[4]
    q_qu = mult_cnstDouble_ct(T[4], qq_divcs_q[5], cryptoContext)
    qq_qu = mult_cnstDouble_ct(T[4], qq_divqr_q[5], cryptoContext)

    for i in range(4):
        # level = T[i] -1
        tmp1 = LevelReduce_ct(T[i], 1)
        tmp = mult_cnstDouble_ct(tmp1, qs_divcs_q[i + 1], cryptoContext)
        q_su = homo_add(q_su, tmp, cryptoContext)
        tmp = mult_cnstDouble_ct(tmp1, qs_divqr_q[i + 1], cryptoContext)
        qs_qu = homo_add(qs_qu, tmp, cryptoContext)

        # level = T[i]
        tmp = mult_cnstDouble_ct(T[i], qq_divcs_q[i + 1], cryptoContext)
        q_qu = homo_add(q_qu, tmp, cryptoContext)

        tmp = mult_cnstDouble_ct(T[i], qq_divqr_q[i + 1], cryptoContext)
        # NOTE: set the last element of qq_divqr_q zero
        qq_qu = homo_add(qq_qu, tmp, cryptoContext)

    qq_qu = rescale_ct(qq_qu, cryptoContext)
    qs_qu = rescale_ct(qs_qu, cryptoContext)

    tmp = LevelReduce_ct(T[5], 1)
    tmp = mult_cnstInt_ct(tmp, (1 << int(qs_lg2_divqr_q_last)), cryptoContext)
    qs_qu = homo_add(qs_qu, tmp, cryptoContext)
    qs_qu = add_cnstDouble_ct(qs_qu, qs_divqr_q[0], cryptoContext)

    tmp= T[5]
    tmp = mult_cnstInt_ct(tmp, (1 << int(qq_lg2_divqr_q_last)), cryptoContext)
    qq_qu = homo_add(qq_qu, tmp, cryptoContext)
    qq_qu = add_cnstDouble_ct(qq_qu, qq_divqr_q[0], cryptoContext)

    q_su = rescale_ct(q_su, cryptoContext)
    q_su = add_cnstDouble_ct(q_su, qs_divcs_q[0], cryptoContext)
    tmp = LevelReduce_ct(T2[1], 1)
    q_su = homo_add(q_su, tmp, cryptoContext)

    q_qu = rescale_ct(q_qu, cryptoContext)
    q_qu = add_cnstDouble_ct(q_qu, qq_divcs_q[0], cryptoContext)
    q_qu = homo_add(q_qu, T2[1], cryptoContext)

    q_qu = homo_mult(q_qu, qq_qu, cryptoContext)
    tmp = LevelReduce_ct(T[4], 1)
    qq_su = mult_cnstDouble_ct(tmp, qq_s2[5], cryptoContext)

    for i in range(4):
        tmp = LevelReduce_ct(T[i], 1)
        tmp = mult_cnstDouble_ct(tmp, qq_s2[i+1], cryptoContext)
        qq_su = homo_add(qq_su, tmp, cryptoContext)

    q_qu = homo_add(q_qu, qq_su, cryptoContext)
    q_qu = rescale_ct(q_qu, cryptoContext)
    tmp = LevelReduce_ct(T[5], 1)
    q_qu = homo_add(q_qu, tmp, cryptoContext)
    q_qu = add_cnstDouble_ct(q_qu, qq_s2[0], cryptoContext)

    # lazy KS: merge KS in computing `q_su` and `qu`
    tmp = LevelReduce_ct(T[4], 1)
    qu = mult_cnstDouble_ct(tmp, q_divcs_q[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = LevelReduce_ct(T[i], 1)
        tmp = mult_cnstDouble_ct(tmp, q_divcs_q[i+1], cryptoContext)
        qu = homo_add(qu, tmp, cryptoContext)
    qu = rescale_ct(qu, cryptoContext)
    qu = add_cnstDouble_ct(qu, q_divcs_q[0], cryptoContext)
    qu = homo_add(qu, T2[2], cryptoContext)

    moduliQ = np.array(cryptoContext.moduliQ, dtype=np.uint64)
    q_mu = np.array(cryptoContext.q_mu, dtype=np.uint64)
    curr_limbs = q_qu.curr_limbs
    tmp_qu = homo_mult_core(qu, q_qu, moduliQ[:curr_limbs], q_mu[:curr_limbs])
    tmp_q_su = homo_mult_core(q_su, qs_qu, moduliQ[:curr_limbs], q_mu[:curr_limbs])
    tmp = LevelReduce_ct(T[4], 2)
    qs_su = mult_cnstDouble_ct(tmp, qs_s2[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = LevelReduce_ct(T[i], 2)
        tmp = mult_cnstDouble_ct(tmp, qs_s2[i+1], cryptoContext)
        qs_su = homo_add(qs_su, tmp, cryptoContext)

    qu = LevelReduce_ct(qu, qu.curr_limbs-curr_limbs)
    qu.cv[0] = polys_add_mod(tmp_qu[0], tmp_q_su[0], moduliQ[:curr_limbs])
    qu.cv[1] = polys_add_mod(tmp_qu[1], tmp_q_su[1], moduliQ[:curr_limbs])
    axax     = polys_add_mod(tmp_qu[2], tmp_q_su[2], moduliQ[:curr_limbs])
    qu.cv[0] = polys_add_mod(qu.cv[0], qs_su.cv[0], moduliQ[:curr_limbs])
    qu.cv[1] = polys_add_mod(qu.cv[1], qs_su.cv[1], moduliQ[:curr_limbs])

    summult = KeySwitch_ct(axax, cryptoContext)
    qu.cv[0] = polys_add_mod(qu.cv[0], summult[0], moduliQ[:curr_limbs])
    qu.cv[1] = polys_add_mod(qu.cv[1], summult[1], moduliQ[:curr_limbs])
    qu.curr_limbs = curr_limbs

    qu = rescale_ct(qu, cryptoContext)
    tmp = LevelReduce_ct(T[5], 2)
    qu = homo_add(qu, tmp, cryptoContext)
    qu = add_cnstDouble_ct(qu, qs_s2[0], cryptoContext)

    return qu

def InnerL1S(T, T2, cryptoContext):
    # weightedSum Core computation
    # level = T[4] -1
    tmp = LevelReduce_ct(T[4], 1)
    q_su = mult_cnstDouble_ct(tmp, ss_divcs_q[5], cryptoContext)
    qs_qu = mult_cnstDouble_ct(tmp, ss_divqr_q[5], cryptoContext)

    # level = T[4]
    q_qu = mult_cnstDouble_ct(T[4], sq_divcs_q[5], cryptoContext)
    qq_qu = mult_cnstDouble_ct(T[4], sq_divqr_q[5], cryptoContext)

    for i in range(4):
        # level = T[i] -1
        tmp1 = LevelReduce_ct(T[i], 1)
        tmp = mult_cnstDouble_ct(tmp1, ss_divcs_q[i + 1], cryptoContext)
        q_su = homo_add(q_su, tmp, cryptoContext)
        tmp = mult_cnstDouble_ct(tmp1, ss_divqr_q[i + 1], cryptoContext)
        qs_qu = homo_add(qs_qu, tmp, cryptoContext)

        # level = T[i]
        tmp = mult_cnstDouble_ct(T[i], sq_divcs_q[i + 1], cryptoContext)
        q_qu = homo_add(q_qu, tmp, cryptoContext)

        tmp = mult_cnstDouble_ct(T[i], sq_divqr_q[i + 1], cryptoContext)
        # NOTE: set the last element of sq_divqr_q zero
        qq_qu = homo_add(qq_qu, tmp, cryptoContext)

    qq_qu = rescale_ct(qq_qu, cryptoContext)
    qs_qu = rescale_ct(qs_qu, cryptoContext)

    tmp = LevelReduce_ct(T[5], 1)
    tmp = mult_cnstInt_ct(tmp, (1 << int(ss_lg2_divqr_q_last)), cryptoContext)
    qs_qu = homo_add(qs_qu, tmp, cryptoContext)
    qs_qu = add_cnstDouble_ct(qs_qu, ss_divqr_q[0], cryptoContext)

    tmp= T[5]
    tmp = mult_cnstInt_ct(tmp, (1 << int(sq_lg2_divqr_q_last)), cryptoContext)
    qq_qu = homo_add(qq_qu, tmp, cryptoContext)
    qq_qu = add_cnstDouble_ct(qq_qu, sq_divqr_q[0], cryptoContext)

    q_su = rescale_ct(q_su, cryptoContext)
    q_su = add_cnstDouble_ct(q_su, ss_divcs_q[0], cryptoContext)
    tmp = LevelReduce_ct(T2[1], 1)
    q_su = homo_add(q_su, tmp, cryptoContext)

    q_qu = rescale_ct(q_qu, cryptoContext)
    q_qu = add_cnstDouble_ct(q_qu, sq_divcs_q[0], cryptoContext)
    q_qu = homo_add(q_qu, T2[1], cryptoContext)

    q_qu = homo_mult(q_qu, qq_qu, cryptoContext)
    tmp = LevelReduce_ct(T[4], 1)
    qq_su = mult_cnstDouble_ct(tmp, sq_s2[5], cryptoContext)

    for i in range(4):
        tmp = LevelReduce_ct(T[i], 1)
        tmp = mult_cnstDouble_ct(tmp, sq_s2[i+1], cryptoContext)
        qq_su = homo_add(qq_su, tmp, cryptoContext)

    q_qu = homo_add(q_qu, qq_su, cryptoContext)
    q_qu = rescale_ct(q_qu, cryptoContext)
    tmp = LevelReduce_ct(T[5], 1)
    q_qu = homo_add(q_qu, tmp, cryptoContext)
    q_qu = add_cnstDouble_ct(q_qu, sq_s2[0], cryptoContext)

    # lazy KS: merge KS in computing `q_su` and `qu`
    tmp = LevelReduce_ct(T[4], 1)
    qu = mult_cnstDouble_ct(tmp, s_divcs_q[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = LevelReduce_ct(T[i], 1)
        tmp = mult_cnstDouble_ct(tmp, s_divcs_q[i+1], cryptoContext)
        qu = homo_add(qu, tmp, cryptoContext)
    qu = rescale_ct(qu, cryptoContext)
    qu = add_cnstDouble_ct(qu, s_divcs_q[0], cryptoContext)
    qu = homo_add(qu, T2[2], cryptoContext)

    moduliQ = np.array(cryptoContext.moduliQ, dtype=np.uint64)
    q_mu = np.array(cryptoContext.q_mu, dtype=np.uint64)
    curr_limbs = q_qu.curr_limbs
    tmp_qu = homo_mult_core(qu, q_qu, moduliQ[:curr_limbs], q_mu[:curr_limbs])
    tmp_q_su = homo_mult_core(q_su, qs_qu, moduliQ[:curr_limbs], q_mu[:curr_limbs])
    tmp = LevelReduce_ct(T[4], 2)
    qs_su = mult_cnstDouble_ct(tmp, ss_s2[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = LevelReduce_ct(T[i], 2)
        tmp = mult_cnstDouble_ct(tmp, ss_s2[i+1], cryptoContext)
        qs_su = homo_add(qs_su, tmp, cryptoContext)

    qu = LevelReduce_ct(qu, qu.curr_limbs-curr_limbs)
    qu.cv[0] = polys_add_mod(tmp_qu[0], tmp_q_su[0], moduliQ[:curr_limbs])
    qu.cv[1] = polys_add_mod(tmp_qu[1], tmp_q_su[1], moduliQ[:curr_limbs])
    axax     = polys_add_mod(tmp_qu[2], tmp_q_su[2], moduliQ[:curr_limbs])
    qu.cv[0] = polys_add_mod(qu.cv[0], qs_su.cv[0], moduliQ[:curr_limbs])
    qu.cv[1] = polys_add_mod(qu.cv[1], qs_su.cv[1], moduliQ[:curr_limbs])

    summult = KeySwitch_ct(axax, cryptoContext)
    qu.cv[0] = polys_add_mod(qu.cv[0], summult[0], moduliQ[:curr_limbs])
    qu.cv[1] = polys_add_mod(qu.cv[1], summult[1], moduliQ[:curr_limbs])
    qu.curr_limbs = curr_limbs

    qu = rescale_ct(qu, cryptoContext)
    tmp = LevelReduce_ct(T[5], 2)
    qu = homo_add(qu, tmp, cryptoContext)
    qu = add_cnstDouble_ct(qu, ss_s2[0], cryptoContext)

    return qu

def EvalChebyshevSeries(x, cryptoContext):
    curr_limbs = x.curr_limbs

    tmp = np.zeros(x.cv.shape, dtype=np.uint64)
    T = [ Ciphertext(tmp, curr_limbs),
        Ciphertext(tmp, curr_limbs),
        Ciphertext(tmp, curr_limbs - 1),
        Ciphertext(tmp, curr_limbs - 2),
        Ciphertext(tmp, curr_limbs - 2),
        Ciphertext(tmp, curr_limbs - 2)]

    # no linear transformation is needed if a = -1, b = 1, T_1(y) = y
    T[0] = x
    # Computes Chebyshev polynomials up to degree k
    # for y: T_1(y) = y, T_2(y), ... , T_k(y)
    # uses binary tree multiplication
    T[1] = homo_square(T[0], cryptoContext)
    T[1] = homo_add(T[1], T[1], cryptoContext)
    T[1] = rescale_ct(T[1], cryptoContext)
    T[1] = add_cnstDouble_ct(T[1], -1.0, cryptoContext)
    T[0] = LevelReduce_ct(T[0], 1)

    T[2] = homo_mult(T[0], T[1], cryptoContext)
    T[2] = homo_add(T[2], T[2], cryptoContext)
    T[2] = rescale_ct(T[2], cryptoContext)
    T[0] = LevelReduce_ct(T[0], 1)
    T[2] = homo_sub(T[2], T[0], cryptoContext)
    T[1] = LevelReduce_ct(T[1], 1)

    T[3] = homo_square(T[1], cryptoContext)
    T[3] = homo_add(T[3], T[3], cryptoContext)
    T[3] = rescale_ct(T[3], cryptoContext)
    T[3] = add_cnstDouble_ct(T[3], -1.0, cryptoContext)

    T[4] = homo_mult(T[1], T[2], cryptoContext)
    T[1] = LevelReduce_ct(T[1], 1)
    T[4] = homo_add(T[4], T[4], cryptoContext)
    T[4] = rescale_ct(T[4], cryptoContext)
    T[0] = LevelReduce_ct(T[0], 1)
    T[4] = homo_sub(T[4], T[0], cryptoContext)

    T[5] = homo_square(T[2], cryptoContext)
    T[2] = LevelReduce_ct(T[2], 1)
    T[5] = homo_add(T[5], T[5], cryptoContext)
    T[5] = rescale_ct(T[5], cryptoContext)
    T[5] = add_cnstDouble_ct(T[5], -1.0, cryptoContext)

    T2 = [ Ciphertext(tmp, curr_limbs),
        Ciphertext(tmp, curr_limbs - 1),
        Ciphertext(tmp, curr_limbs - 2),
        Ciphertext(tmp, curr_limbs - 3)]

    # Compute the Chebyshev polynomials T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    T2[1] = homo_square(T[5], cryptoContext)
    T[5] = LevelReduce_ct(T[5], 1)
    T2[1] = homo_add(T2[1], T2[1], cryptoContext)
    T2[1] = rescale_ct(T2[1], cryptoContext)
    T2[1] = add_cnstDouble_ct(T2[1], -1.0, cryptoContext)

    # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
    tmpct2=T2[1]
    T2km1 = T[5]
    tmpct2 = homo_add(tmpct2, tmpct2, cryptoContext)
    tmpct2 = add_cnstDouble_ct(tmpct2, -1.0, cryptoContext)
    T2km1 = homo_mult(T2km1, tmpct2, cryptoContext)
    T2km1 = rescale_ct(T2km1, cryptoContext)

    for i in range(2,4):
        square = homo_square(T2[i-1], cryptoContext)
        T2[i] = homo_add(square, square, cryptoContext)
        T2[i] = rescale_ct(T2[i], cryptoContext)
        T2[i] = add_cnstDouble_ct(T2[i], -1.0, cryptoContext)

        # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        tmpct2= T2[i]
        T2km1 = homo_mult(T2km1, tmpct2, cryptoContext)
        T2km1 = homo_add(T2km1, T2km1, cryptoContext)
        T2km1 = rescale_ct(T2km1, cryptoContext)
        tmpct2 = T[5]
        tmpct2 = LevelReduce_ct(tmpct2, i)
        T2km1 = homo_sub(T2km1, tmpct2, cryptoContext)

    qu = InnerL1Q(T, T2, cryptoContext)
    su = InnerL1S(T, T2, cryptoContext)

    result = mult_cnstDouble_ct(T[0], first_divcs_q[1], cryptoContext)
    for i in range(1, 5):
        tmp = mult_cnstDouble_ct(T[i], first_divcs_q[i+1], cryptoContext)
        result = homo_add(result, tmp, cryptoContext)

    result = LevelReduce_ct(result, 2)
    result = rescale_ct(result, cryptoContext)

    result = add_cnstDouble_ct(result, first_divcs_q[0], cryptoContext)
    result = homo_add(result, T2[3], cryptoContext)

    result = homo_mult(result, qu, cryptoContext)
    result = rescale_ct(result, cryptoContext)
    su = LevelReduce_ct(su, 1)
    result = homo_add(result, su, cryptoContext)
    result = homo_sub(result, T2km1, cryptoContext)

    return result

def DoubleAngleIteration(in0, cryptoContext):
    r = int(R)
    scalar=[-0.94418452709144784,-0.89148442119890103,-0.79474447324033948,-0.63161877774606467,-0.3989422804014327,-0.15915494309189535,]

    for j in range(1,r+1):
        in0 = homo_square(in0, cryptoContext)
        in0 = ModReduce_ct(in0, 1, cryptoContext)
        in0 = homo_add(in0, in0, cryptoContext)
        # scalar = np.float64(np.float64(-1.0) / np.float64(math.pow((2.0 * M_PI), np.float64(math.pow(2.0, j - r)))))
        in0 = add_cnstDouble_ct(in0, scalar[j-1], cryptoContext)

    return in0
