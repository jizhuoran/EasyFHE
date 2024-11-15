import numpy as np
import time
from enum import Enum

import torch
import torch.fhe.functional as F
import torch.fhe.test as T
import torch.fhe.testBig as TB
from torch.fhe.context import Context

import torch.fhe.homo_ops as OP

import torch.fhe.client.client as client

param, cc, key = client.gen_crypto_context(3, 50, 8)
mult_key = client.gen_mult_keys(cc, key)
rotate_key = client.gen_rotate_keys(cc, key, [1, -2])

x1 = torch.tensor([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])
print("x1: ", x1)
cipher1 = client.encrypt(x1, cc, key)

print("cipher1: ", cipher1)

plain1 = client.decrypt(cipher1, param, cc, key)
print("plain1: ", plain1)


exit(0)

TB.test_HMult3()

TB.test_ApproxMod()