import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import homo_ops
from . import KeySwitch
# from .data import Hmult3_N16_L12_K3 as HMult3
from .data import params_ks_13 as N8192KS
from .data import cheby_N16 as chebyData
Tensor = torch.Tensor

# my_dict = {
#     'moduliQ12_N65536' : HMult3.moduliQ12_N65536,
#     'moduliP3_N65536' : HMult3.moduliP3_N65536,
#     'rootsQ12_N65536' : HMult3.rootsQ12_N65536,
#     'rootsP3_N65536' : HMult3.rootsP3_N65536,
#     'swk' : HMult3.swk,
#     'cipher1_' : HMult3.cipher1_,
#     'axax0' : HMult3.axax0,
#     'sumMult0' : HMult3.sumMult0,
#     'cipher1_mult1' : HMult3.cipher1_mult1,
#     'cipher1_mult1_rescale1' : HMult3.cipher1_mult1_rescale1,
#     'axax1' : HMult3.axax1,
#     'sumMult1' : HMult3.sumMult1,
#     'cipher1_mult2_rescale1' : HMult3.cipher1_mult2_rescale1,
#     'cipher1_mult2_rescale2' : HMult3.cipher1_mult2_rescale2,
#     'axax2' : HMult3.axax2,
#     'sumMult2' : HMult3.sumMult2,
#     'cipher1_mult3_rescale2' : HMult3.cipher1_mult3_rescale2,
#     'cipher1_mult3_rescale3' : HMult3.cipher1_mult3_rescale3
# }

# np.savez('torch/fhe/data/Hmult3_N16_L12_K3', **my_dict)  

class Hmult3_N16_L12_K3:
    def __init__(self):
        my_dict = np.load('torch/fhe/data/Hmult3_N16_L12_K3.npz', allow_pickle=True)
        self.moduliQ12_N65536 = my_dict["moduliQ12_N65536"]
        self.moduliP3_N65536 = my_dict["moduliP3_N65536"]
        self.rootsQ12_N65536 = my_dict["rootsQ12_N65536"]
        self.rootsP3_N65536 = my_dict["rootsP3_N65536"]
        self.swk = my_dict["swk"]
        self.cipher1_ = my_dict["cipher1_"]
        self.axax0 = my_dict["axax0"]
        self.sumMult0 = my_dict["sumMult0"]
        self.cipher1_mult1 = my_dict["cipher1_mult1"]
        self.cipher1_mult1_rescale1 = my_dict["cipher1_mult1_rescale1"]
        self.axax1 = my_dict["axax1"]
        self.sumMult1 = my_dict["sumMult1"]
        self.cipher1_mult2_rescale1 = my_dict["cipher1_mult2_rescale1"]
        self.cipher1_mult2_rescale2 = my_dict["cipher1_mult2_rescale2"]
        self.axax2 = my_dict["axax2"]
        self.sumMult2 = my_dict["sumMult2"]
        self.cipher1_mult3_rescale2 = my_dict["cipher1_mult3_rescale2"]
        self.cipher1_mult3_rescale3 = my_dict["cipher1_mult3_rescale3"]

HMult3 = Hmult3_N16_L12_K3()


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




def test_ApproxMod():
    #test case: N64_L18_P1
    logN = 16
    N = 2**logN
    L = 18
    K = 1
    moduliQ = chebyData.moduliQ18_N65536
    moduliP = chebyData.moduliP1_N65536
    rootsQ = chebyData.rootsQ18_N65536
    rootsP = chebyData.rootsP1_N65536
    dnum = int(L / K)
    # swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk = chebyData.swk
    swk = swk.reshape((2, dnum, L + K, N))

    cryptoContext = Context(logN,
                            60, 59, 60,
                            L, K,
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    print("finish gen cryptoContext")

    curr_limbs=L-2
    cv = np.array(chebyData.cheby_input, dtype=np.uint64)
    cv = cv.reshape(2, curr_limbs, N)
    ct = Ciphertext(cv, curr_limbs)
    print("finish gen Ciphertext")


    res = homo_ops.EvalChebyshevSeries(ct, cryptoContext)
    print("finish EvalChebyshev")

    golden_answer = np.array(chebyData.cheby_output, dtype=np.uint64)
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
    print("finish DoubleAngleIteration")

    golden_answer = np.array(chebyData.doubleAngle_output, dtype=np.uint64)
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

