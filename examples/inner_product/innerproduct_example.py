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
import os
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


def homo_inner_product_hoisting(cipher_A, cipher_B, cryptoContext):
    cipher_product = fhe.homo_mul(cipher_A, cipher_B, cryptoContext)
    cipher_product = fhe.homo_rescale(cipher_product, 1, cryptoContext)
    n = cipher_product.slots
    assert (n & (n - 1)) == 0, "n must be a power of two"
    log_n = int(math.log2(n))
    shifts = [2 ** i for i in range(log_n)]


    bxExt = fhe.key_switch_P_ext(fhe.extract_cv(cipher_product,0, cryptoContext), cryptoContext)
    ax = fhe.extract_cv(cipher_product,1, cryptoContext)
    for shift in shifts:
        # ver1
        # rotated = fhe.homo_rotate(cipher_product, shift, cryptoContext)

        # ver2
        # rotated_modup = fhe.modup_to_ext(fhe.extract_cv(cipher_product, 1, cryptoContext), cryptoContext)
        # rotated_innerp = fhe.mult_rot_key_and_sum_ext(rotated_modup, shift, cryptoContext)
        # rotated = fhe.moddown_from_ext(rotated_innerp, cryptoContext)
        # rotated.cv[0] = F.cv_add(rotated.cv[0], cipher_product.cv[0], cryptoContext.moduliQ, rotated.cur_limbs)
        # rotated = fhe.cipher_automorphism(rotated, cryptoContext.norm_rot_index(shift), cryptoContext, printInfo=False)

        # ver3
        rotated_modup = fhe.modup_to_ext(ax, cryptoContext)
        norm_index = cryptoContext.norm_rot_index(shift)
        rotated_innerp = fhe.mult_rot_key_and_sum_ext(rotated_modup, norm_index, cryptoContext)

        tmp_bxExt = fhe.homo_add(fhe.extract_cv(rotated_innerp,0, cryptoContext), bxExt, cryptoContext)
        tmp_ax    = fhe.moddown_from_ext(fhe.extract_cv(rotated_innerp,1, cryptoContext), cryptoContext)

        tmp_bxExt = fhe.cipher_automorphism(tmp_bxExt, norm_index, cryptoContext)
        tmp_ax  = fhe.cipher_automorphism(tmp_ax, norm_index, cryptoContext)

        bxExt = fhe.homo_add(tmp_bxExt, bxExt, cryptoContext)
        ax    = fhe.homo_add(tmp_ax, ax, cryptoContext)

    bx = fhe.moddown_from_ext(bxExt, cryptoContext)
    cipher_product = cipher_product.cipher_like([bx.cv[0], ax.cv[0]])

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

config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, SAVE_MIDDLE=False)
cryptoContext, openfhe_context = (
    fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                         levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=DATA_DIR, config=config))


values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
encode_slots = (1 << 11)
x = np.array([values[i % len(values)] for i in range(encode_slots)])
x = torch.tensor(x, device="cuda")
cipher = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

values1 = [0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888]
x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
cipher1 = openfhe_context.encrypt(x1, 1, openfhe_context.depth - 1, encode_slots)
x1 = torch.tensor(x1, device="cuda")

cipher_inner_product = homo_inner_product(cipher,cipher1,cryptoContext)
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
print("HE decryption result: ", clear_result[0], "\n\n")


cipher_inner_product_hoisting = homo_inner_product_hoisting(cipher,cipher1,cryptoContext)
torch.cpu.synchronize()
torch.cuda.synchronize()
start_time = time.time()
cipher_inner_product_hoisting = homo_inner_product_hoisting(cipher,cipher1,cryptoContext)
torch.cpu.synchronize()
torch.cuda.synchronize()
print("time: ", time.time() - start_time)
print("homo_inner_product done!")
print("homo_inner_product_hoisting done!")
clear_result = openfhe_context.decrypt(cipher_inner_product_hoisting)
clear_result = clear_result.cpu().numpy().reshape(-1)
print("HE decryption result: ", clear_result[0])

print("plain",plain_inner_product(x, x1))
