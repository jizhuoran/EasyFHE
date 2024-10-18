import math

import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import functional as F
from . import homo_ops
from . import arithmetic
from . import KeySwitch
from .data import params_N256_L4_P1 as N256L4P1
from .data import params_N256_L4_P2 as N256L4P2
from .data import params_N64 as N64
from .data import params_N64_cheby as N64_cheby
from .data import params_ks_13 as N8192KS
from .data import params_ks_17 as N131072KS
Tensor = torch.Tensor


def compare_and_print(res, golden, test_name):
    compare = np.array_equal(res, golden)
    # compare = res == golden_answer
    print(f"\ntest {test_name}: \nresult: ")
    print(compare)

def test_KS_ModDown2():
    # test case 1
    axax = N256L4P1.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 1
    moduliQ = N256L4P1.moduliQ4_N256
    moduliP = N256L4P1.moduliP1_N256
    rootsQ = N256L4P1.rootsQ4_N256
    rootsP = N256L4P1.rootsP1_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P1.swk_bxL4P1
    for i in range(dnum):
        swk[0, i] = N256L4P1.swk_axL4P1
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)

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

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                               QHatInvModq, moduliP, pInvVec, pRootScalePows, QHatModp, L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)

    res_0 = KeySwitch.ModDown_method2(sumMult[0], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                      pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, L, K, N)

    res_1 = KeySwitch.ModDown_method2(sumMult[1], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                      pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, L, K, N)

    golden_answer = N256L4P1.res[0]
    golden_answer = golden_answer.reshape(res_0.shape)
    compare = np.array_equal(res_0, golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 1: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P1.res[1]
    golden_answer = golden_answer.reshape(res_1.shape)
    compare = np.array_equal(res_1, golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

    #test case 2
    axax = N256L4P2.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)

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

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
                               moduliP, pInvVec, pRootScalePows, QHatModp,
                               L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)
    res_0 = KeySwitch.ModDown_method2(sumMult[0],
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    res_1 = KeySwitch.ModDown_method2(sumMult[1],
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    golden_answer = N256L4P2.res[0]
    golden_answer = golden_answer.reshape(res_0.shape)
    compare = np.array_equal(res_0, golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P2.res[1]
    golden_answer = golden_answer.reshape(res_1.shape)
    compare = np.array_equal(res_1, golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_KS_components():
    # test case 1
    axax = N256L4P1.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 1
    moduliQ = N256L4P1.moduliQ4_N256
    moduliP = N256L4P1.moduliP1_N256
    rootsQ = N256L4P1.rootsQ4_N256
    rootsP = N256L4P1.rootsP1_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P1.swk_bxL4P1
    for i in range(dnum):
        swk[0, i] = N256L4P1.swk_axL4P1
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)

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

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                               QHatInvModq, moduliP, pInvVec, pRootScalePows, QHatModp, L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)

    res_0 = KeySwitch.ModDown(sumMult[0], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, L, K, N)

    res_1 = KeySwitch.ModDown(sumMult[1], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, L, K, N)

    golden_answer = N256L4P1.res[0]
    golden_answer = golden_answer.reshape(res_0.shape)
    compare = np.array_equal(res_0, golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 1: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P1.res[1]
    golden_answer = golden_answer.reshape(res_1.shape)
    compare = np.array_equal(res_1, golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")


    #test case 2
    axax = N256L4P2.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)

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

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
                               moduliP, pInvVec, pRootScalePows, QHatModp,
                               L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)
    res_0 = KeySwitch.ModDown(sumMult[0],
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    res_1 = KeySwitch.ModDown(sumMult[1],
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    golden_answer = N256L4P2.res[0]
    golden_answer = golden_answer.reshape(res_0.shape)
    compare = np.array_equal(res_0, golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P2.res[1]
    golden_answer = golden_answer.reshape(res_1.shape)
    compare = np.array_equal(res_1, golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_KS_ModDown_ct():
    #test case N256_L4_P2
    axax = N256L4P2.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)

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

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
                               moduliP, pInvVec, pRootScalePows, QHatModp,
                               L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)
    res = KeySwitch.ModDown_ct(sumMult,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    golden_answer = N256L4P2.res[0]
    golden_answer = golden_answer.reshape(res[0].shape)
    compare = np.array_equal(res[0], golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P2.res[1]
    golden_answer = golden_answer.reshape(res[1].shape)
    compare = np.array_equal(res[1], golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_KS_ct():
    #test case N256_L4_P2
    axax = N256L4P2.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)

    mult_swk = cryptoContext.mult_swk
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
                                   QHatModp, NScaleInvModp, pHatInvModp, L, K, N)

    golden_answer = N256L4P2.res[0]
    golden_answer = golden_answer.reshape(res[0].shape)
    compare = np.array_equal(res[0], golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P2.res[1]
    golden_answer = golden_answer.reshape(res[1].shape)
    compare = np.array_equal(res[1], golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")


def test_KS3_ct():
    logN = 13
    N = 2**logN
    L = 4
    K = 2
    moduliQ = N8192KS.moduliQ4_N8192
    moduliP = N8192KS.moduliP2_N8192
    rootsQ = N8192KS.rootsQ4_N8192
    rootsP = N8192KS.rootsP2_N8192
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk = N8192KS.swk
    swk = swk.reshape(2, dnum, L + K, N)
    print(swk.shape)
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)

    mult_swk = cryptoContext.mult_swk
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

    ########## test L=4 ##########
    axax = N8192KS.axax0
    axax= axax.reshape((4,8192))
    print(axax.shape)
    res = KeySwitch.KeySwitch_core(axax, mult_swk, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                   QHatInvModq, pHatModq, PInvModq, moduliP, pInvVec, pRootScalePows, pRootScalePowsInv,
                                   QHatModp, NScaleInvModp, pHatInvModp, L, K, N)

    golden_answer = N8192KS.sumMult0[0]
    golden_answer = golden_answer.reshape(res[0].shape)
    compare = np.array_equal(res[0], golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 1 L = 4: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N8192KS.sumMult0[1]
    golden_answer = golden_answer.reshape(res[1].shape)
    compare = np.array_equal(res[1], golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

    ########## test L=3 ##########
    axax = N8192KS.axax1
    axax = axax.reshape((3, 8192))

    curr_limbs= L-1
    beta = int(math.ceil(curr_limbs / K))  # total beta groups
    ceil_curr_limbs = int(beta) * K
    tmp = np.random.randint(0, 2 ** 50, size=(ceil_curr_limbs-curr_limbs, N), dtype=np.uint64)

    axax_pad = np.vstack((axax, tmp)) # dont move, must be declared here
    print(axax_pad.shape)

    res = KeySwitch.KeySwitch_core(axax, mult_swk, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv,
                                       NScaleInvModq,
                                       QHatInvModq, pHatModq, PInvModq, moduliP, pInvVec, pRootScalePows,
                                       pRootScalePowsInv,
                                       QHatModp, NScaleInvModp, pHatInvModp, curr_limbs, K, N)

    golden_answer = N8192KS.sumMult1[0]
    golden_answer = golden_answer.reshape(res[0].shape)
    compare = np.array_equal(res[0], golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2 L = 3: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N8192KS.sumMult1[1]
    golden_answer = golden_answer.reshape(res[1].shape)
    compare = np.array_equal(res[1], golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

    ########## test padding KS ##########
    QHatModp_pad = cryptoContext.PartQlHatModp_pad  # note! param for padding
    res_pad = KeySwitch.KeySwitch_core_pad(axax_pad, mult_swk, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                       QHatInvModq, pHatModq, PInvModq, moduliP, pInvVec, pRootScalePows, pRootScalePowsInv,
                                       QHatModp_pad, NScaleInvModp, pHatInvModp, curr_limbs, K, N)
    res_pad = np.delete(res_pad, range(curr_limbs,ceil_curr_limbs), axis=1) # tailor the output
    print(res_pad.shape)# 输出结果形状
    compare = res_pad == res
    print("\n\ntest padding KS: \n")
    print(compare)
    print("\n")


def test_logN17():
    axax = N131072KS.axax1
    axax = axax.reshape((3, 131072))
    print(axax.shape)
    logN = 17
    N = 2 ** logN
    L = 4
    K = 2
    moduliQ = N131072KS.moduliQ4_N131072
    moduliP = N131072KS.moduliP2_N131072
    rootsQ = N131072KS.rootsQ4_N131072
    rootsP = N131072KS.rootsP2_N131072
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk = N131072KS.swk
    swk = swk.reshape(2, dnum, L + K, N)
    print(swk.shape)
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)

    mult_swk = cryptoContext.mult_swk
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

    res = KeySwitch.KeySwitch_core(axax, mult_swk, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv,
                                   NScaleInvModq,
                                   QHatInvModq, pHatModq, PInvModq, moduliP, pInvVec, pRootScalePows,
                                   pRootScalePowsInv,
                                   QHatModp, NScaleInvModp, pHatInvModp, L-1, K, N)

    golden_answer = N131072KS.sumMult1[0]
    golden_answer = golden_answer.reshape(res[0].shape)
    compare = np.array_equal(res[0], golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N131072KS.sumMult1[1]
    golden_answer = golden_answer.reshape(res[1].shape)
    compare = np.array_equal(res[1], golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_SwitchModulus():
    print("------------------")
    print("test_SwitchModulus")
    print("------------------")
    # N=64
    ax = N64.ct0_before_SwitchModulus
    bx = N64.ct1_before_SwitchModulus
    golden_ax = N64.ct0_after_SwitchModulus
    golden_bx = N64.ct1_after_SwitchModulus
    res_ax = arithmetic.vec_switch_modulus(ax, 36028797018918401, 1152921504606844417)
    res_bx = arithmetic.vec_switch_modulus(bx, 36028797018918401, 1152921504606844417)
    compare_and_print(res_ax, golden_ax, "res_ax")
    compare_and_print(res_bx, golden_bx, "res_bx")

def test_homo_add():
    print("-------------")
    print("test_homo_add")
    print("-------------")
    L=2
    K=1
    a = torch.tensor([[[1, 2, 3, 4], [4, 5, 6, 4]],[[7, 8, 9, 4], [0, 2, 0, 4]]], dtype=torch.uint64, device='cuda')
    b = torch.tensor([[[4, 5, 6, 4], [4, 5, 6, 4]],[[4, 5, 6, 4], [4, 5, 6, 4]]], dtype=torch.uint64, device='cuda')
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    in0 = Ciphertext(a, L)
    in1 = Ciphertext(b, L)
    cryptoContext = Context(2, 53, 52, 52, L, K)
    cryptoContext.moduliQ = np.array([10,11], dtype=np.uint64)
    ct_ = homo_ops.homo_add(in0, in1, cryptoContext)
    golden_answer = torch.tensor([[[ 5,  7,  9,  8], [ 8, 10,  1,  8]], [[ 1,  3,  5,  8], [ 4,  7, 6, 8]]], dtype = torch.uint64)
    ct_.cv = torch.from_numpy(ct_.cv)
    correct = torch.equal(golden_answer, ct_.cv)
    # print(ct_.cv)
    print("test_homo_add")
    print(correct)
    print("\n")


def test_HMultCore():
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2

    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    ct = Ciphertext(cv, L)
    q_mu = cryptoContext.q_mu
    res = homo_ops.homo_mult_core(ct, ct, moduliQ, q_mu)

    golden_answer = np.array(N256L4P2.cipher1_after_multCore, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res[:2].shape)

    compare = np.array_equal(res[0], golden_answer[0])
    # compare = res == golden_answer
    print("test_HMultCore")
    print("\n\ntest 2: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res[1], golden_answer[1])
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res[2], N256L4P2.axax_)
    # compare = res == golden_answer
    print("\nres_cx result: ")
    print(compare)
    print("\n")

def test_HMult():

    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2

    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    ct = Ciphertext(cv, L)
    res = homo_ops.homo_mult(ct, ct, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)

    compare = np.array_equal(res.cv[0], golden_answer[0])
    # compare = res == golden_answer
    print("\n\ntest 2: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res.cv[1], golden_answer[1])
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_rescale_ct():
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_after_mult, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)
    ct = Ciphertext(cv, L)
    res = homo_ops.rescale_ct(ct, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult_rescale, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)
    compare = np.array_equal(res.cv[0], golden_answer[0])
    print("\n\ntest 2: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res.cv[1], golden_answer[1])
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_HMult_and_rescale_1():
    print("------------------------")
    print("test_HMult_and_rescale_1")
    print("------------------------")
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2

    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    ct = Ciphertext(cv, L)
    mult = homo_ops.homo_mult(ct, ct, cryptoContext)
    res = homo_ops.rescale_ct(mult, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult_rescale, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)

    compare = np.array_equal(res.cv[0], golden_answer[0])
    # compare = res == golden_answer
    print("test 2: \n\nres_ax result: ")
    print(compare)

    compare = np.array_equal(res.cv[1], golden_answer[1])
    # compare = res == golden_answer
    print("res_bx result: ")
    print(compare)
    print("\n")

def test_ModReduce_ct():
    print("------------------")
    print("test_ModReduce_ct")
    print("------------------")
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_after_mult, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)
    ct = Ciphertext(cv, L)
    res = homo_ops.ModReduce_ct(ct, 1, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult_ModReduce, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)
    compare = np.array_equal(res.cv[0], golden_answer[0])
    print("test 2: \n\nres_ax result: ")
    print(compare)

    compare = np.array_equal(res.cv[1], golden_answer[1])
    print("res_bx result: ")
    print(compare)

def test_ApproxMod():
    #test case: N64_L18_P1
    logN = 6
    N = 2**logN
    L = 18
    K = 1
    moduliQ = N64_cheby.moduliQ18_N64
    moduliP = N64_cheby.moduliP1_N64
    rootsQ = N64_cheby.rootsQ18_N64
    rootsP = N64_cheby.rootsP1_N64
    dnum = int(L / K)
    # swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk = N64_cheby.swk
    swk = swk.reshape((2, dnum, L + K, N))

    cryptoContext = Context(logN,
                            60, 59, 60,
                            L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)

    L=L-2
    cv = np.array(N64_cheby.cheby_input, dtype=np.uint64)
    cv = cv.reshape(2, L, N)

    ct = Ciphertext(cv, L)
    res = homo_ops.EvalChebyshevSeries(ct, cryptoContext)

    golden_answer = np.array(N64_cheby.cheby_output, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)

    compare = np.array_equal(res.cv[0], golden_answer[0])
    # compare = res == golden_answer
    print("\n\ntest cheby: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res.cv[1], golden_answer[1])
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

    res = homo_ops.DoubleAngleIteration(res, cryptoContext)
    golden_answer = np.array(N64_cheby.doubleAngle_output, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)

    compare = np.array_equal(res.cv[0], golden_answer[0])
    # compare = res == golden_answer
    print("\n\ntest doubleAngle: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res.cv[1], golden_answer[1])
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_cuda_KS():
    axax = N131072KS.axax0
    axax = axax.reshape((4, 131072))
    curr_limbs = 4
    logN = 17
    N = 2 ** logN
    L = 4
    K = 2
    moduliQ = N131072KS.moduliQ4_N131072
    moduliP = N131072KS.moduliP2_N131072
    rootsQ = N131072KS.rootsQ4_N131072
    rootsP = N131072KS.rootsP2_N131072
    dnum = int(L / K)
    swk = N131072KS.swk
    swk = swk.reshape(2, dnum, L + K, N)
    context_cuda = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    
    input_ks = torch.tensor(axax.reshape(-1), dtype=torch.uint64, device="cuda")
    ax = torch.tensor(swk[0].reshape(-1),dtype=torch.uint64,device="cuda")
    bx = torch.tensor(swk[1].reshape(-1),dtype=torch.uint64,device="cuda")
    res = F.keyswitch(context_cuda=context_cuda, 
                      input = input_ks,
                      swk_ax=ax,
                      swk_bx =bx,
                      curr_limbs =curr_limbs)
    
    res0 = res[0].detach().cpu().numpy()
    res1 = res[1].detach().cpu().numpy()

    golden_answer = N131072KS.sumMult0[0]
    golden_answer = golden_answer.reshape(res0.shape)
    compare = np.array_equal(res0, golden_answer)
    print("\n\ntest: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N131072KS.sumMult0[1]
    golden_answer = golden_answer.reshape(res1.shape)
    compare = np.array_equal(res1, golden_answer)
    print("\nres_bx result: ")
    print(compare)
    print("\n")
        