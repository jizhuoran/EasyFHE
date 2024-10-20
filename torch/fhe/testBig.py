import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import homo_ops
from . import KeySwitch
from .data import Hmult3_N16_L12_K3 as HMult3
from .data import params_ks_13 as N8192KS
Tensor = torch.Tensor

def compare_and_print(res, golden, test_name):
    # compare = np.array_equal(res, golden)
    # # compare = res == golden_answer
    # print(f"\ntest {test_name}: \nresult: ")
    # print(compare)

    compare0 = np.array_equal(res[0], golden[0])
    compare1 = np.array_equal(res[1], golden[1])
    # compare = res == golden_answer
    print(f"\ntest {test_name}: \nresult: ")
    print(compare0)
    print(compare1)

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

    golden_answer = np.array(N8192KS.sumMult0, dtype=np.uint64).reshape(res.shape)
    compare_and_print(res, golden_answer, "l=4,KS1")

    ########## test L=3 ##########
    axax = N8192KS.axax1
    axax = axax.reshape((3, 8192))
    curr_limbs= L-1

    res = KeySwitch.KeySwitch_core(axax, mult_swk, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv,
                                       NScaleInvModq,
                                       QHatInvModq, pHatModq, PInvModq, moduliP, pInvVec, pRootScalePows,
                                       pRootScalePowsInv,
                                       QHatModp, NScaleInvModp, pHatInvModp, curr_limbs, K, N)
    golden_answer = np.array(N8192KS.sumMult1, dtype=np.uint64).reshape(res.shape)
    compare_and_print(res, golden_answer, "l=3,KS2")



def test_HMult3():
    print("------------------------")
    print("test_HMult_and_rescale_1")
    print("------------------------")
    logN = 16
    N = 2**logN
    L = 12
    K = 3
    cv = np.array(HMult3.cipher1_, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = HMult3.moduliQ12_N65536
    moduliP = HMult3.moduliP3_N65536
    rootsQ = HMult3.rootsQ12_N65536
    rootsP = HMult3.rootsP3_N65536
    dnum = int(L / K)

    swk = HMult3.swk
    swk = swk.reshape(2, dnum, L + K, N)

    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    ct = Ciphertext(cv, L)

    mult = homo_ops.homo_mult(ct, ct, cryptoContext)
    golden_answer = np.array(HMult3.cipher1_mult1, dtype=np.uint64).reshape(mult.cv.shape)
    compare_and_print(mult.cv, golden_answer, "mult1")

    res = homo_ops.ModReduce_ct(mult, 1, cryptoContext)
    golden_answer = np.array(HMult3.cipher1_mult1_rescale1, dtype=np.uint64).reshape(res.cv.shape)
    compare_and_print(res.cv, golden_answer, "mult1_rescale1")

    mult = homo_ops.homo_mult(res, res, cryptoContext)
    golden_answer = np.array(HMult3.cipher1_mult2_rescale1, dtype=np.uint64).reshape(mult.cv.shape)
    compare_and_print(mult.cv, golden_answer, "mult2_rescale1")

    res = homo_ops.ModReduce_ct(mult, 1, cryptoContext)
    golden_answer = np.array(HMult3.cipher1_mult2_rescale2, dtype=np.uint64).reshape(res.cv.shape)
    compare_and_print(res.cv, golden_answer, "mult2_rescale2")

    mult = homo_ops.homo_mult(res, res, cryptoContext)
    golden_answer = np.array(HMult3.cipher1_mult3_rescale2, dtype=np.uint64).reshape(mult.cv.shape)
    compare_and_print(mult.cv, golden_answer, "mult3_rescale1")

    res = homo_ops.ModReduce_ct(mult, 1, cryptoContext)
    golden_answer = np.array(HMult3.cipher1_mult3_rescale3, dtype=np.uint64).reshape(res.cv.shape)
    compare_and_print(res.cv, golden_answer, "mult3_rescale3")


