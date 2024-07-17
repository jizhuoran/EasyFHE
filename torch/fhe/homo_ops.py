import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import functional as F
from . import arithmetic
from . import KeySwitch
Tensor = torch.Tensor

def polys_add_mod(a, b, MOD):
    assert a.shape == b.shape and a.shape[0] == len(MOD)
    c_ = torch.zeros(a.shape, dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a).cuda()
    b_ = torch.from_numpy(b).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_add_mod(a_[i], b_[i], val)
    c = c_.cpu().numpy()
    return c

def polys_sub_mod(a, b, MOD):
    assert a.shape == b.shape and a.shape[0] == len(MOD)
    c_ = torch.zeros(a.shape, dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a).cuda()
    b_ = torch.from_numpy(b).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_sub_mod(a_[i], b_[i], val)
    c = c_.cpu().numpy()
    return c

def polys_mul_mod(a, b, MOD, mu):
    assert a.shape == b.shape and a.shape[0] == len(MOD)
    c_ = torch.zeros(a.shape, dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a).cuda()
    b_ = torch.from_numpy(b).cuda()
    mu_ = torch.from_numpy(mu).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_mul_mod(a_[i], b_[i], val, mu_[i])
    c = c_.cpu().numpy()
    return c

def homo_add(in0, in1, cryptoContext):
    assert in0.curr_limbs == in1.curr_limbs
    curr_limbs = in0.curr_limbs
    MOD = cryptoContext.moduliQ[:curr_limbs]
    res = np.zeros(in0.cv.shape, dtype=np.uint64)
    res[0] = polys_add_mod(in0.cv[0], in1.cv[0], MOD)  # res.ax
    res[1] = polys_add_mod(in0.cv[1], in1.cv[1], MOD)  # res.bx

    return Ciphertext(res, curr_limbs)

def homo_mult_core(in0, in1, moduliQ, q_mu):
    assert in0.curr_limbs == in1.curr_limbs
    curr_limbs = in0.curr_limbs
    MOD = moduliQ[:curr_limbs]
    MU = q_mu[:curr_limbs]
    res = np.zeros((3, in0.cv.shape[1], in0.cv.shape[2]), dtype=np.uint64)

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
    assert in0.curr_limbs == in1.curr_limbs

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

    res = homo_mult_core(in0, in1, moduliQ, q_mu)
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

    res = np.zeros((2, ct.curr_limbs - 1, N), dtype=np.uint64)
    for l in range(levels):
        for k in range(2):
            res[k] = DropLastElementAndScale(
                ct.cv[k], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                QlQlInvModqlDivqlModq[diffQl + l], qInvModq,
                curr_limbs, N)
    return Ciphertext(res, ct.curr_limbs-1)

def LevelReduce_ct(ct, levels):
    assert ct.curr_limbs > 1 and ct.curr_limbs > levels
    ct.cv = ct.cv[:, :ct.curr_limbs-levels, :]
    return ct
