import numpy as np
import torch
import torch.fhe.functional as F
import torch.fhe.test as T
from torch.fhe.Ciphertext import Ciphertext
from torch.fhe.context import Context

T.test_homo_add()
T.test_HMult_and_rescale_1()
T.test_SwitchModulus()