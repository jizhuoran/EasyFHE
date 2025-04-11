##########################
#### example for app #####
##########################
import torch.fhe as fhe
import numpy as np
import torch
import warnings,os

maxLevelsRemaining = 12
appRotIndex_list = [-1, 2]
logBsSlots_list = [11, 12]
logN = 14
dnum = 3
dcrtBits = 52
firstMod = 56
levelBudget_list = [[3, 3], [4, 4]]
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"

DATA_DIR = os.environ["DATA_DIR"]

config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, COMPARE_WITH_OPENFHE=False)
cryptoContext, openfhe_context = (
    fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                         levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=DATA_DIR,
                         config=config))

values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
encode_slots = (1 << 11)
x = np.array([values[i % len(values)] for i in range(encode_slots)])
x = torch.tensor(x, device="cuda")
cipher = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

values1 = [0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888]
x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
ptx = fhe.encode(x1, "x1", 0, encode_slots, cryptoContext)

# do some application computation
cipher = fhe.homo_rotate(cipher, -1, cryptoContext)
cipher = fhe.homo_rotate(cipher, 2, cryptoContext)
print("homo_rotate done!")

# bootstrapping
result = fhe.homo_bootstrap(cipher, cryptoContext.L, logBsSlots_list[0], cryptoContext)
print("gpu bootstrapp done!")

clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
clear_result = clear_result.cpu().numpy().reshape(-1)
print("HE decryption result: ", clear_result[:10])

# do some application computation
approx_plain_val = clear_result[:10]
# print(approx_plain_val)
for i in range(result.cur_limbs - 4):
    approx_plain_val = approx_plain_val * values1[0]
    # print(approx_plain_val)
    result = fhe.homo_mul_pt(result, ptx, cryptoContext)
    result = fhe.homo_rescale(result,1, cryptoContext)  # for FIXEDMANUAL mode only

# do another bootstrapping
result1 = fhe.homo_bootstrap(result, cryptoContext.L, logBsSlots_list[1], cryptoContext)
print("gpu bootstrapp done!")

clear_result = openfhe_context.decrypt(result1)  # decrypt by cc with different slots value should be fine
clear_result = clear_result.cpu().numpy().reshape(-1)
warnings.warn(
    "Note: OpenFHE adds random noise during decryption. As a result, the decrypted output may vary slightly each time, "
    "and may differ from the OpenFHE decryption result even within the same round."
)
print("plain result: ", approx_plain_val)
print("HE decryption result: ", clear_result[:10])

is_equal = np.allclose(clear_result[:10], approx_plain_val[:10], atol=1e-02)
# compare elements of clear_result and approx_plain_eval, if absolute distance is less then 1e-03, then is equal
if is_equal:
    print("app: Test passed!")
else:
    print("app: Test failed!")