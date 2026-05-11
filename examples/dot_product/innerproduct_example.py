##########################
#### example for app #####
##########################
import sys, os, time
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))

import easyfhe.fhe as fhe
import numpy as np
import easyfhe as torch
import os
import math

@fhe.utils.profile_python_function
def homo_inner_product(cipher_A, cipher_B, n, cryptoContext):
    assert (n & (n - 1)) == 0, "n must be a power of two"
    cipher_product = fhe.homo_mul(cipher_A, cipher_B, cryptoContext)
    cipher_product = fhe.homo_rescale(cipher_product, 1, cryptoContext)
    log_n = int(math.log2(n))
    shifts = [2**i for i in range(log_n)] 
    for shift in shifts:
        rotated = fhe.homo_rotate(cipher_product, shift, cryptoContext)
        cipher_product = fhe.homo_add(cipher_product, rotated, cryptoContext)
    
    return cipher_product


def plain_inner_product(x, x1):
    return torch.dot(x, x1)

encode_slots = (1 << 9)
print("encode_slots:", encode_slots)
maxLevelsRemaining = 1
appRotIndex_list = [1, 2, 3, 4, 5, 6, 7,
                    8, 16, 24, 32,
                    64, 96, 128,
                    256, 384, 512,
                    1024
                    ]


logBsSlots_list = []
logN = 16
dnum = 3
dcrtBits = 58
firstMod = 60
levelBudget_list = []
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
device = "cuda"

DATA_DIR = os.environ["DATA_DIR"]

config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, SAVE_MIDDLE=False)
cryptoContext, openfhe_context = (
    fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                         levelBudget_list, "UNIFORM_TERNARY", rescaleTech, device, save_dir=DATA_DIR, config=config))

values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

x = np.array([values[i % len(values)] for i in range(encode_slots)])
cipher = openfhe_context.encrypt(x, cryptoContext.device, 1, openfhe_context.depth - 1, encode_slots)
x = torch.tensor(x, device="cuda")

values1 = [0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111]
x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
cipher1 = openfhe_context.encrypt(x1, cryptoContext.device, 1, openfhe_context.depth - 1, encode_slots)
x1 = torch.tensor(x1, device="cuda")

for n in [8, 32, 128, 512]:
    print("vector length for inner product operations", n, "\n")

    cipher_inner_product = homo_inner_product(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    start_time = time.time()
    cipher_inner_product = homo_inner_product(cipher,cipher1, n, cryptoContext)
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    print("time: ", time.time() - start_time)
    print("homo_inner_product done!")
    clear_result = openfhe_context.decrypt(cipher_inner_product)
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[0], "\n\n")

    print("plain",plain_inner_product(x, x1))

    print("\n\n")
