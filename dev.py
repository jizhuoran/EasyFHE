import numpy as np
# import time
# from enum import Enum

# import torch
# import torch.fhe.functional as F
# import torch.fhe.test as T
# import torch.fhe.testBig as TB
import torch.fhe.bootstrapping as bstest
# from torch.fhe.context import Context

# import torch.fhe.homo_ops as OP

import torch.fhe.client.client as client


# cryptoContext.moduliQ = torch.tensor(cryptoContext.moduliQ, dtype=torch.uint64, device="cuda")
# cryptoContext.q_mu = torch.tensor(cryptoContext.q_mu, dtype=torch.uint64, device="cuda")
#
# rotate_key = client.gen_rotate_keys(cc, key, [1, -2])
#
# x1 = torch.tensor([0.25, 0.5, 0.75], device="cuda")
# print("x1: ", x1)
# cipher1 = client.encrypt(x1, cc, key)
# cipher2 = OP.homo_mul(cipher1, cipher1, cryptoContext)
# cipher2 = OP.cipher_rescale(cipher2, cryptoContext)
#
# plain2 = client.decrypt(cipher2, param, cc, key)
# print("plain2: ", plain2)
#
# cipher3 = OP.homo_mul(cipher2, cipher2, cryptoContext)
# cipher3 = OP.cipher_rescale(cipher3, cryptoContext)
# plain3 = client.decrypt(cipher3, param, cc, key)
#
# print("plain3: ", plain3)


bstest.run_test_cases()
# bstest.BootstrapTest_N65536L26lB44()

# TB.test_HMult3()

# TB.test_ApproxMod()