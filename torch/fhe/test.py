import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import functional as F
from . import homo_ops
from .data import params_N256_L4_P1 as N256L4P1
from .data import params_N256_L4_P2 as N256L4P2
from .data import params_N64 as N64
Tensor = torch.Tensor


def test_homo_add():
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