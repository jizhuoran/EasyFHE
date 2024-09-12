from . import arithmetic
import numpy as np
import math

def Decompose(axax, curr_limbs, K, N):
    beta = int(math.ceil(curr_limbs / K))  # total beta groups
    d2Tilde = np.zeros((beta, curr_limbs + K, N), dtype=np.uint64)

    for j in range(beta):
        start = K * j
        end = min(start + K, curr_limbs)
        d2Tilde[j, start:end, :] = axax[start:end, :]

    return d2Tilde

def InnerProduct(d2Tilde, key,
                 moduliQ, moduliP,
                 curr_limbs, K, N):

    sumMult = np.zeros((2, curr_limbs + K, N), dtype=np.uint64)
    product = np.zeros((2, curr_limbs + K, N), dtype=np.uint64)
    L = len(moduliQ)
    beta = math.ceil(curr_limbs / K)

    for k in range(2): # deal with 2 dim of a ct
        for j in range(beta):
            for i in range(curr_limbs):
                product[k][i] = arithmetic.vec_mul_mod(d2Tilde[j][i], key[k][j][i], moduliQ[i])
                sumMult[k][i] = arithmetic.vec_add_mod(sumMult[k][i], product[k][i], moduliQ[i])
            for i, pi in zip(range(curr_limbs, curr_limbs + len(moduliP)), range(len(moduliP))):
                product[k][i] = arithmetic.vec_mul_mod(d2Tilde[j][i], key[k][j][L+pi], moduliP[pi])
                sumMult[k][i] = arithmetic.vec_add_mod(sumMult[k][i], product[k][i], moduliP[pi])

    return sumMult


def ModUp(a, d2Tilde,
          moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
          moduliP, pInvVec, pRootScalePows, QHatModp,
          curr_limbs, K, N):

    intt_a = np.zeros((curr_limbs, N), np.uint64)
    beta = math.ceil(curr_limbs / K)

    for i in range(curr_limbs):
        intt_a[i] = arithmetic.iNTT(a[i], N, moduliQ[i], qInvVec[i], qRootScalePowsInv[i], NScaleInvModq[i])


    arithmetic.ModUp_Core(intt_a, d2Tilde,
               moduliQ, moduliP, QHatInvModq, QHatModp,
               curr_limbs, K, N)

    moduliQP = np.concatenate((moduliQ[:curr_limbs], moduliP))
    qpInv = np.concatenate((qInvVec[:curr_limbs], pInvVec))
    qpRSP = np.concatenate((qRootScalePows[:curr_limbs], pRootScalePows))
    for j in range(beta):
        in_C_L_index = j * K
        in_C_L_len = K if j < (beta - 1) else (curr_limbs - in_C_L_index)

        ranges = list(range(0, in_C_L_index)) + list(range(in_C_L_index + in_C_L_len, curr_limbs + K))
        for i in ranges:
            d2Tilde[j][i] = arithmetic.NTT(d2Tilde[j][i], N, moduliQP[i], qpInv[i], qpRSP[i])

    return d2Tilde


def ModDown(a,
            moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
            moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
            curr_limbs, K, N):

    intt_a = np.zeros((curr_limbs+K, N), np.uint64)

    moduliQP = np.concatenate((moduliQ[:curr_limbs], moduliP))
    qpInv = np.concatenate((qInvVec[:curr_limbs], pInvVec))
    qpRSPInv = np.concatenate((qRootScalePowsInv[:curr_limbs], pRootScalePowsInv))
    qpNScaleInvMod = np.concatenate((NScaleInvModq[:curr_limbs], NScaleInvModp))
    for i in range(curr_limbs+K):
        intt_a[i] = arithmetic.iNTT(a[i], N, moduliQP[i], qpInv[i], qpRSPInv[i], qpNScaleInvMod[i])


    res = arithmetic.ModDown_Core(intt_a,
                 pHatInvModp, pHatModq, PInvModq,
                 moduliQ, moduliP,
                 N, curr_limbs, K,)

    for i in range(curr_limbs):
        res[i] = arithmetic.NTT(res[i], N, moduliQ[i], qInvVec[i], qRootScalePows[i])

    return res

def ModDown_method2(a,
            moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
            moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
            curr_limbs, K, N):

    intt_a = np.zeros((curr_limbs+K, N), np.uint64)

    moduliQP = np.concatenate((moduliQ[:curr_limbs], moduliP))
    qpInv = np.concatenate((qInvVec[:curr_limbs], pInvVec))
    qpRSPInv = np.concatenate((qRootScalePowsInv[:curr_limbs], pRootScalePowsInv))
    qpNScaleInvMod = np.concatenate((NScaleInvModq[:curr_limbs], NScaleInvModp))
    for i in range(curr_limbs, curr_limbs+K):
        intt_a[i] = arithmetic.iNTT(a[i], N, moduliQP[i], qpInv[i], qpRSPInv[i], qpNScaleInvMod[i])

    res = np.zeros((curr_limbs, N), dtype=np.uint64)
    tmp3 = np.zeros((K, N), dtype=np.uint64)
    tmpk = intt_a[curr_limbs:, :]
    for k in range(K):
        tmp3[k] = arithmetic.vec_mul_scalar_mod(tmpk[k], pHatInvModp[k], moduliP[k])

    for i in range(curr_limbs):
        sum = [int(0) for _ in range(N)]
        for k in range(K):
            product = arithmetic.vec_mul_scalar_int(tmp3[k], pHatModq[k][i])
            sum = arithmetic.vec_add_int(sum, product)

        res[i] = arithmetic.vec_mod_int(sum, moduliQ[i])
        res[i] = arithmetic.NTT(res[i], N, moduliQ[i], qInvVec[i], qRootScalePows[i])

        res[i] = arithmetic.vec_sub_mod(a[i], res[i], moduliQ[i])
        res[i] = arithmetic.vec_mul_scalar_mod(res[i], PInvModq[i], moduliQ[i])


    return res


def ModDown_ct(input,
               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
               curr_limbs, K, N):
    res = np.zeros((2, curr_limbs, N), np.uint64)
    res[0] = ModDown(input[0], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, curr_limbs, K, N)

    res[1] = ModDown(input[1], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, curr_limbs, K, N)
    return res

def KeySwitch_core(axax, swk,
                 moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq, pHatModq, PInvModq,
                 moduliP, pInvVec, pRootScalePows, pRootScalePowsInv, QHatModp, NScaleInvModp, pHatInvModp,
                 curr_limbs, K, N):

    d2Tilde = Decompose(axax, curr_limbs, K, N)
    d2Tilde = ModUp(axax, d2Tilde,
                    moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
                    moduliP, pInvVec, pRootScalePows, QHatModp,
                    curr_limbs, K, N)
    sumMult = InnerProduct(d2Tilde, swk,
                           moduliQ, moduliP,
                           curr_limbs, K, N)
    res = ModDown_ct(sumMult,
                     moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                     moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                     curr_limbs, K, N)

    return res