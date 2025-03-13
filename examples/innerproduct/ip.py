##########################
#### example for app #####
##########################
import sys, os, time
import numpy as np

sys.path.append("/".join(os.getcwd().split("/")[:-2]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))

import torch.fhe as fhe
import numpy as np
import torch
import warnings,os
import math


def homo_inner_product(cipher_A, cipher_B, cryptoContext):
    cipher_product = fhe.homo_mul(cipher_A, cipher_B, cryptoContext)
    cipher_product = fhe.homo_rescale(cipher_product, 1, cryptoContext)
    n = cipher_product.slots
    assert (n & (n - 1)) == 0, "n must be a power of two"
    log_n = int(math.log2(n))
    shifts = [2**i for i in range(log_n)] 
    for shift in shifts:
        rotated = fhe.homo_rotate(cipher_product, shift, cryptoContext)
        cipher_product = fhe.homo_add(cipher_product, rotated, cryptoContext)
    
    return cipher_product

def plain_inner_product(x, x1):
    return torch.dot(x, x1)


maxLevelsRemaining = 5
appRotIndex_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
logBsSlots_list = []
logN = 16
dnum = 3
dcrtBits = 48 
firstMod = 60
levelBudget_list = []
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
mode = "release"  # "debug" or "release"
autoLoadAndSetConfig = True # note: currently only support True

DATA_DIR = os.environ["DATA_DIR"]

cryptoContext, openfhe_context = (
    fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                         levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=DATA_DIR,
                         autoLoadAndSetConfig=True, mode=mode))


values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
encode_slots = (1 << 11)
x = np.array([values[i % len(values)] for i in range(encode_slots)])
x = torch.tensor(x, device="cuda")
cipher = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots, mode)

values1 = [0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888]
x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
x1 = torch.tensor(x1, device="cuda")
ptx = fhe.encode(x1, 1, 0, encode_slots, use_gpu_fft=True, cryptoContext=cryptoContext)
cipher1 = openfhe_context.encrypt(x1, 1, openfhe_context.depth - 1, encode_slots, mode)

# clear_result = openfhe_context.decrypt(cipher1)  
# clear_result = clear_result.cpu().numpy().reshape(-1)
# print("HE decryption result: ", clear_result[0])
# inner_product
torch.cpu.synchronize()
torch.cuda.synchronize()
start_time = time.time()
cipher_inner_product = homo_inner_product(cipher,cipher1,cryptoContext)
torch.cpu.synchronize()
torch.cuda.synchronize()
print("time: ", time.time() - start_time)
print("homo_inner_product done!")

clear_result = openfhe_context.decrypt(cipher_inner_product)  
clear_result = clear_result.cpu().numpy().reshape(-1)
print("HE decryption result: ", clear_result[0])
# print("plain",plain_inner_product(x, x1))

