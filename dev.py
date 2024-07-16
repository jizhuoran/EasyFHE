import numpy as np
import torch
import torch.fhe.functional as F
import torch.fhe.test as T
from torch.fhe.Ciphertext import Ciphertext
from torch.fhe.context import Context

# a = torch.tensor([1, 2, 6], dtype=torch.uint64, device='cuda')
# b = torch.tensor([4, 5, 6], dtype=torch.uint64, device='cuda')
# # x = 2**128//mod
# # low = low64(x)
# # high = high64(x)
# mu = torch.tensor([14347467612885206812, 2049638230412172401], dtype=torch.uint64, device='cuda')
# c = F.vec_mul_mod(a, b, 9, mu)
# print(c)

T.test_HMultCore()
T.test_homo_add()