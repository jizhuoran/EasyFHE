import numpy as np
import time
from enum import Enum

import torch
import torch.fhe.functional as F
import torch.fhe.test as T
import torch.fhe.testBig as TB
from torch.fhe.Ciphertext import Ciphertext
from torch.fhe.context import Context

# TB.test_HMult3()

TB.test_HMult3()
# TB.test_ApproxMod()

TB.test_ApproxMod()