################################
#### example for bootstrap #####
################################

import os, sys, warnings

sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch.fhe as fhe
import numpy as np
import torch

DATA_DIR = os.environ["DATA_DIR"]

maxLevelsRemaining = 12
appRotIndex_list = [-1, 2]
logBsSlots_list = [11, 12]
logN = 14
levelBudget_list = [[3, 3], [4, 4]]
save_dir = DATA_DIR
autoLoadAndSetConfig = True  # note: currently only support True

if not os.path.exists(save_dir):
    raise ValueError(f"Directory {save_dir} does not exist!")

cryptoContext, openfhe_context = fhe.try_load_context(
    maxLevelsRemaining,
    appRotIndex_list,
    logBsSlots_list,
    logN,
    levelBudget_list,
    save_dir=save_dir,
    autoLoadAndSetConfig=True,
)

values = [
    0.111111,
    0.222222,
    0.333333,
    0.444444,
    0.555555,
    0.666666,
    0.777777,
    0.888888,
]
encode_slots = 1 << 11
x = np.array([values[i % len(values)] for i in range(encode_slots)])
x = torch.tensor(x, device="cuda")
cipher = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

values1 = [
    0.888888,
    0.888888,
    0.888888,
    0.888888,
    0.888888,
    0.888888,
    0.888888,
    0.888888,
]
x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
x1 = torch.tensor(x1, device="cuda")
ptx = openfhe_context.encode(x1, 1, 0, encode_slots)

# do some application computation
cipher = fhe.homo_rotate(cipher, -1, cryptoContext)
cipher = fhe.homo_rotate(cipher, 2, cryptoContext)
print("homo_rotate done!")

# bootstrapping
result = fhe.homo_bootstrap(
    cipher,
    L0=cryptoContext.L,
    logBsSlots=logBsSlots_list[0],
    cryptoContext=cryptoContext,
)
print("gpu bootstrapp done!")

clear_result = openfhe_context.decrypt(result)
clear_result = clear_result.cpu().numpy().reshape(-1)
print("HE decryption result: ", clear_result[:10])

# do some application computation
approx_plain_val = clear_result[:10]
for i in range(result.cur_limbs - 4):
    approx_plain_val = approx_plain_val * values1[0]
    result = fhe.homo_mul_pt(result, ptx, cryptoContext)

# do another bootstrapping
result1 = fhe.homo_bootstrap(
    result,
    L0=cryptoContext.L,
    logBsSlots=logBsSlots_list[1],
    cryptoContext=cryptoContext,
)
print("gpu bootstrapp done!")

clear_result = openfhe_context.decrypt(result1)
clear_result = clear_result.cpu().numpy().reshape(-1)
print("plain result: ", approx_plain_val)
print("HE decryption result: ", clear_result[:10])

is_equal = np.allclose(clear_result[:10], approx_plain_val[:10], atol=1e-02)
if is_equal:
    print("app: Test passed!")
else:
    print("app: Test failed!")
