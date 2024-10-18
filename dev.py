import numpy as np
import time
from enum import Enum

import torch
import torch.fhe.functional as F
import torch.fhe.test as T
from torch.fhe.Ciphertext import Ciphertext
from torch.fhe.context import Context
# from torch.fhe.context_cuda import Context_Cuda

# T.test_homo_add()
# T.test_HMult_and_rescale_1()
# T.test_SwitchModulus()
# T.test_ApproxMod()
# T.test_logN17()
T.test_cuda_KS()
# T.test_KS3_ct()
# a = torch.tensor([6] * (2**15), dtype=torch.uint64, device='cuda')
# b = torch.tensor([4] * (2**15), dtype=torch.uint64, device='cuda')
#
# mu = torch.tensor([14347467612885206812, 2049638230412172401], dtype=torch.uint64, device='cuda')

# c = F.vec_mul_mod(a, 7, 9, mu)
