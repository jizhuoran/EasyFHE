import torch.fhe.example.dev_test as dev_test
#
# dev_test.app_example_debug(mode="debug")
dev_test.app_example_release(mode="release")
# dev_test.encode_test_case(mode="debug")
# dev_test.ct_pt_test_case(mode="debug")

# import torch.fhe as fhe
# import numpy as np
# import torch
# import warnings,os
# maxLevelsRemaining = 3,
# appRotIndex_list = [-1, 2],
# logSlots_list = [11, 12],
# logN = 14,
# dnum = 3,
# dcrtBits = 52,
# firstMod = 56,
# levelBudget_list = [[3, 3], [4, 4]],
# approxModDepth = 9,
# rescaleTech = "FLEXIBLEAUTO",  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
# save_dir = "torch/fhe/data/",
# mode = "release"  # "debug" or "release"
#
# if not os.path.exists(save_dir):
#     raise ValueError(f"Directory {save_dir} does not exist!")
#
# cryptoContext, openfhe_context = (
#     fhe.try_load_context(maxLevelsRemaining, appRotIndex_list, logSlots_list, logN,
#                            dnum, dcrtBits, firstMod, levelBudget_list, approxModDepth,
#                            "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir, mode=mode))
#
# specify_slots = logSlots_list[0]  # logslots = 11
# values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
# x = np.array([values[i % len(values)] for i in range((1 << specify_slots))])
# x = torch.tensor(x, device="cuda")
# cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1 << specify_slots)
#
# values1 = [0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888]
# x1 = np.array([values1[i % len(values1)] for i in range((1 << specify_slots))])
# x1 = torch.tensor(x1, device="cuda")
# cipher1, cipher1_openfhe = openfhe_context.encrypt(x1, 1, 0, 1 << specify_slots)
#
# # do the application computation
# utils.load_rotation_keys(cryptoContext, "app")
# cipher = fhe.homo_rotate(cipher, -1, cryptoContext)
# cipher = fhe.homo_rotate(cipher, 2, cryptoContext)
# print("homo_rotate done!")
#
# # bootstrapping, logSlots = 11
# cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
# cryptoContext.BsContext.to_cuda()
# utils.load_rotation_keys(cryptoContext, specify_slots)
#
# result = fhe.homo_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
# print("gpu bootstrapp done!")
#
# clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
# clear_result = clear_result.cpu().numpy().reshape(-1)
# print("HE decryption result: ", clear_result[:10])  # should be of len 10
#
# # #####################################
# # # ..., omit some homomorphic computation
# # #####################################
#
# # # bootstrapping, logSlots = 12
# result.slots = (1 << specify_slots)  # This assignment is for testing purposes only.
# cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
# cryptoContext.BsContext.to_cuda()
# utils.load_rotation_keys(cryptoContext, specify_slots)
#
# approx_plain_val = clear_result[:10]
# # print(approx_plain_val)
# for i in range(result.cur_limbs - 4):
#     approx_plain_val = approx_plain_val * values1[0]
#     # print(approx_plain_val)
#     result = fhe.homo_mul(result, cipher1, cryptoContext)
#
# result1 = fhe.homo_bootstrap(result, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
# print("gpu bootstrapp done!")
#
# clear_result = openfhe_context.decrypt(result1)  # decrypt by cc with different slots value should be fine
# clear_result = clear_result.cpu().numpy().reshape(-1)
# # print # note!!! openfhe给解密加了随机噪声，所以openfhe和gpu相同的多项式但是两边解密结果不一样！
# warnings.warn(
#     "note: openfhe adds random noise during decryption, therefore the result might be slightly different each time, "
#     "and might be different from the openfhe decryption result even in the same round")
# print("plain result: ", approx_plain_val)
# print("HE decryption result: ", clear_result[:10])  # should be of len 10
#
# is_equal = np.allclose(clear_result[:10], approx_plain_val[:10], atol=1e-02)
# # compare elements of clear_result and approx_plain_eval, if absolute distance is less then 1e-03, then is equal
# if is_equal:
#     print("app: Test passed!")
# else:
#     print("app: Test failed!")