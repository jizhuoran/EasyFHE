import torch.fhe.example.dev_test as dev_test
# note: the following test may not use the same ctx, therefore should not run in one round
dev_test.app_without_bs_example_debug(mode="debug")
dev_test.app_example_debug(mode="debug")
dev_test.app_example_release(mode="release")
dev_test.encode_test_case(mode="debug")
dev_test.ct_pt_test_case(mode="debug")


##########################
#### example for app #####
##########################
import torch.fhe as fhe
import numpy as np
import torch
import warnings,os

maxLevelsRemaining = 3
appRotIndex_list = [-1, 2]
logBsSlots_list = [11, 12]
logN = 14
dnum = 3
dcrtBits = 52
firstMod = 56
levelBudget_list = [[3, 3], [4, 4]]
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
save_dir = "torch/fhe/data/"
mode = "release"  # "debug" or "release"

if not os.path.exists(save_dir):
    raise ValueError(f"Directory {save_dir} does not exist!")

cryptoContext, openfhe_context = (
    fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                         levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir, mode=mode))


values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
encode_slots = (1 << 11)
x = np.array([values[i % len(values)] for i in range(encode_slots)])
x = torch.tensor(x, device="cuda")
cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

values1 = [0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888]
x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
x1 = torch.tensor(x1, device="cuda")
cipher1, cipher1_openfhe = openfhe_context.encrypt(x1, 1, 0, encode_slots)

# do the application computation
fhe.load_rotation_keys(cryptoContext, "app")
# todo: assign specific rot_index_list or load all the keys
# todo: add offload and set for rotation keys and bs context separately
cipher = fhe.homo_rotate(cipher, -1, cryptoContext)
cipher = fhe.homo_rotate(cipher, 2, cryptoContext)
print("homo_rotate done!")

# bootstrapping
fhe.load_bootstrapping_context(logBsSlots_list[0],
                               cryptoContext)  # logBsSlots = 11, lb = [3,3] #todo: the online load is for performance only?
result = fhe.homo_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots_list[0], cryptoContext=cryptoContext)
print("gpu bootstrapp done!")

clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
clear_result = clear_result.cpu().numpy().reshape(-1)
print("HE decryption result: ", clear_result[:10])

# #####################################
# # ..., omit some homomorphic computation
# #####################################

# # bootstrapping, logBsSlots = 12, lb = [4,4]
encode_slots = (1<<12)
result.slots = encode_slots  # This assignment is for testing purposes only.
fhe.load_bootstrapping_context(logBsSlots_list[1], cryptoContext)
#todo: load, offload, set: set_bootstrapping_keys(he_res20_ctx.cur_num_slots, cryptoContext)

approx_plain_val = clear_result[:10]
# print(approx_plain_val)
for i in range(result.cur_limbs - 4):
    approx_plain_val = approx_plain_val * values1[0]
    # print(approx_plain_val)
    result = fhe.homo_mul(result, cipher1, cryptoContext)

result1 = fhe.homo_bootstrap(result, L0=cryptoContext.L, logBsSlots=logBsSlots_list[1], cryptoContext=cryptoContext)
print("gpu bootstrapp done!")

clear_result = openfhe_context.decrypt(result1)  # decrypt by cc with different slots value should be fine
clear_result = clear_result.cpu().numpy().reshape(-1)
# print # note!!! openfhe给解密加了随机噪声，所以openfhe和gpu相同的多项式但是两边解密结果不一样！
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