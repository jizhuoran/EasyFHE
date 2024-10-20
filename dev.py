# import numpy as np
# import time
# from enum import Enum

# import torch
# import torch.fhe.functional as F
# import torch.fhe.test as T
import torch.fhe.testBig as TB
# from torch.fhe.Ciphertext import Ciphertext
# from torch.fhe.context import Context


# from torch.fhe.context_cuda import Context_Cuda

# from .torch.fhe.data import params_N256_L4_P2 as N256L4P2

# logN = 16
# N = 2**logN
# L = 4
# K = 2

# cv = np.zeros((2, L, N), dtype=np.uint64)
# cv = cv.reshape(2, L, N)
# ct = Ciphertext(cv, L)

# mod = np.ones(L, dtype=np.uint64)

# x = ct
# y = ct

# moduliQ = np.ones(L, dtype=np.uint64)
# moduliP = np.ones(L, dtype=np.uint64)
# rootsQ = np.ones(L, dtype=np.uint64)
# rootsP = np.ones(L, dtype=np.uint64)
# dnum = int(L / K)
# cryptoContext = Context(logN, 53, 52, 52, L, K, moduliQ, moduliP, rootsQ, rootsP)



# z = F.cv_add_scalar(x, y, cryptoContext)
# print(z)

# T.test_homo_add()
# T.test_HMult_and_rescale_1()
# T.test_SwitchModulus()
# T.test_ApproxMod()
# T.test_cuda_KS()
# TB.test_KS3_ct()
TB.test_HMult3()
# T.test_logN17()
# a = torch.tensor([6] * (2**15), dtype=torch.uint64, device='cuda')
# b = torch.tensor([4] * (2**15), dtype=torch.uint64, device='cuda')
#
# mu = torch.tensor([14347467612885206812, 2049638230412172401], dtype=torch.uint64, device='cuda')

# c = F.vec_mul_mod(a, 7, 9, mu)
