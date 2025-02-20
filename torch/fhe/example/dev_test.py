from ..bs_context import *
from .. import homo_ops
from .. import utils
from ..bootstrapping import eval_bootstrap
import numpy as np
import os, warnings

def app_example_debug(
        logN=14,
        logSlots_list=[11, 12],
        maxLevelsRemaining=3,
        levelBudget_list=[[3, 3], [4, 4]],
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        approxModDepth=9,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="torch/fhe/data/",
        mode = "debug" # "debug" or "release"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    cryptoContext, openfhe_context_dict = utils.try_load_context(logN,
                                                                 logSlots_list,
                                                                 maxLevelsRemaining,
                                                                 levelBudget_list,
                                                                 dnum,
                                                                 dcrtBits,
                                                                 firstMod,
                                                                 approxModDepth,
                                                                 [-1,2],
                                                                 "UNIFORM_TERNARY",
                                                                 rescaleTech,
                                                                 save_dir=save_dir,
                                                                 mode = mode)

    specify_slots = logSlots_list[0] # logslots = 11
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<specify_slots)

    # do the application computation
    utils.load_rotation_keys(cryptoContext, "app")
    cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    print("homo_rotate done!")
    # compute golden answer
    if mode == "debug":
        cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe, -1)
        cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe,2)
        is_euqal = utils.compare_bs_ct_with_openfhe(cipher, cipher_openfhe)
        if is_euqal:
            print("homo_rotate: Test passed!")
        else:
            print("homo_rotate: Test failed!")

    # bootstrapping, logSlots = 11
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)
    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    print("gpu bootstrapp done!")
    # compute golden answer
    if mode == "debug":
        cipher_openfhe.SetSlots((1<<specify_slots))
        openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
        is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
        if is_euqal:
            print("BootstrapTest_logslots11: Test passed!")
        else:
            print("BootstrapTest_logslots11: Test failed!")

    # #####################################
    # # ..., omit some homomorphic computation
    # #####################################

    # bootstrapping, logSlots = 12
    specify_slots = logSlots_list[1]
    openfhe_context1 = openfhe_context_dict[str(specify_slots)]
    result.slots = (1<<specify_slots) # This assignment is for testing purposes only.
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)
    for i in range(result.cur_limbs - 3):
        result = homo_ops.homo_mul(result, result, cryptoContext)
        openfhe_boot = openfhe_context.cc.EvalSquare(openfhe_boot)
    result1 = eval_bootstrap(result, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    print("gpu bootstrapp done!")
    # compute golden answer
    if mode == "debug":
        openfhe_boot.SetSlots((1 << specify_slots)) # to cheat openfhe boot with (1<<specify_slots)
        openfhe_boot1 = openfhe_context1.cc.EvalBootstrap(openfhe_boot)
        is_euqal = utils.compare_bs_ct_with_openfhe(result1, openfhe_boot1)
        if is_euqal:
            print("BootstrapTest_logslots12: Test passed!")
        else:
            print("BootstrapTest_logslots12: Test failed!")


def app_example_release(
        logN=14,
        logSlots_list=[11, 12],
        maxLevelsRemaining=3,
        levelBudget_list=[[3, 3], [4, 4]],
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        approxModDepth=9,
        rescaleTech="FLEXIBLEAUTO",  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="torch/fhe/data/",
        mode="release"  # "debug" or "release"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    app_rot_index_list = [-1, 2]
    cryptoContext, openfhe_context = utils.try_load_context(logN,
                                                            logSlots_list,
                                                            maxLevelsRemaining,
                                                            levelBudget_list,
                                                            dnum,
                                                            dcrtBits,
                                                            firstMod,
                                                            approxModDepth,
                                                            app_rot_index_list,
                                                            "UNIFORM_TERNARY",
                                                            rescaleTech,
                                                            save_dir=save_dir,
                                                            mode=mode)

    specify_slots = logSlots_list[0]  # logslots = 11
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1 << specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1 << specify_slots)

    values1 = [0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888]
    x1 = np.array([values1[i % len(values1)] for i in range((1 << specify_slots))])
    x1 = torch.tensor(x1, device="cuda")
    cipher1, cipher1_openfhe = openfhe_context.encrypt(x1, 1, 0, 1 << specify_slots)

    # do the application computation
    utils.load_rotation_keys(cryptoContext, "app")
    cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    print("homo_rotate done!")

    # bootstrapping, logSlots = 11
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)

    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    print("gpu bootstrapp done!")

    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[:10])  # should be of len 10

    # #####################################
    # # ..., omit some homomorphic computation
    # #####################################

    # # bootstrapping, logSlots = 12
    result.slots = (1 << specify_slots)  # This assignment is for testing purposes only.
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)


    approx_plain_val = clear_result[:10]
    # print(approx_plain_val)
    for i in range(result.cur_limbs - 4):
        approx_plain_val = approx_plain_val * values1[0]
        # print(approx_plain_val)
        result = homo_ops.homo_mul(result, cipher1, cryptoContext)

    result1 = eval_bootstrap(result, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    print("gpu bootstrapp done!")

    clear_result = openfhe_context.decrypt(result1)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    # print # note!!! openfhe给解密加了随机噪声，所以openfhe和gpu相同的多项式但是两边解密结果不一样！
    warnings.warn("note: openfhe adds random noise during decryption, therefore the result might be slightly different each time, "
          "and might be different from the openfhe decryption result even in the same round")
    print("plain result: ", approx_plain_val)
    print("HE decryption result: ", clear_result[:10])  # should be of len 10

    is_equal = np.allclose(clear_result[:10], approx_plain_val[:10], atol=1e-03)
    # compare elements of clear_result and approx_plain_eval, if absolute distance is less then 1e-03, then is equal
    if is_equal:
        print("app: Test passed!")
    else:
        print("app: Test failed!")


def encode_test_case(
        logN=14,
        logSlots_list=[11],
        maxLevelsRemaining=3,
        levelBudget_list=[[3, 3]],
        dnum=3,
        dcrtBits=59,
        firstMod=60,
        approxModDepth=9,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="torch/fhe/data/",
        mode = "debug" # "debug" or "release"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    cryptoContext, openfhe_context_dict = utils.try_load_context(logN,
                                                                 logSlots_list,
                                                                 maxLevelsRemaining,
                                                                 levelBudget_list,
                                                                 dnum,
                                                                 dcrtBits,
                                                                 firstMod,
                                                                 approxModDepth,
                                                                 [],
                                                                 "UNIFORM_TERNARY",
                                                                 rescaleTech,
                                                                 save_dir=save_dir,
                                                                 mode = mode)

    specify_slots = logSlots_list[0] # logslots = 11
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    x = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])
    plaintext        = openfhe_context.encode_gpu_fhe(cryptoContext, x)
    plaintext_golden = openfhe_context.encode(x)

    encoded_data = plaintext.mv[0]
    encoded_data = encoded_data.cpu().numpy()
    encode_data_golden = plaintext_golden.mv[0]
    encode_data_golden = encode_data_golden.cpu().numpy()
    all_correct = True
    for i in range(len(encode_data_golden)):
        diff_indices = np.where(encode_data_golden[i] != encoded_data[i])
        if len(diff_indices[0]) > 0:
            print("diff_indices: ", diff_indices[0][:10])
            print("len(diff_indices): ", len(diff_indices[0]))
            all_correct = False
            if i == 0:  # prt a wrong case
                print(encode_data_golden[0][:10])
                print(encoded_data[0][:10])

    if (plaintext==plaintext_golden) != True:
        all_correct = False
        print("ground_truth: ")
        print(plaintext_golden.cur_limbs)
        print(plaintext_golden.noise_deg)
        print(plaintext_golden.scaling_factor)
        print("result: ")
        print(plaintext.cur_limbs)
        print(plaintext.noise_deg)
        print(plaintext.scaling_factor)

    if all_correct:
        print("all_correct for this test")

    print("done")

def ct_pt_test_case(
        logN=14,
        logSlots_list=[8],
        maxLevelsRemaining=3,
        levelBudget_list=[[4, 4]],
        dnum=3,
        dcrtBits=59,
        firstMod=60,
        approxModDepth=9,
        # rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        rescaleTech = "FIXEDMANUAL", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="torch/fhe/data/",
        mode = "debug" # "debug" or "release"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    cryptoContext, openfhe_context_dict = utils.try_load_context(logN,
                                                                 logSlots_list,
                                                                 maxLevelsRemaining,
                                                                 levelBudget_list,
                                                                 dnum,
                                                                 dcrtBits,
                                                                 firstMod,
                                                                 approxModDepth,
                                                                 [],
                                                                 "UNIFORM_TERNARY",
                                                                 rescaleTech,
                                                                 save_dir=save_dir,
                                                                 mode = mode)

    specify_slots = logSlots_list[0] # logslots = 11
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1 << specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, 0,
                                                     (1 << specify_slots))
    encoded = openfhe_context.encode(values, 1, 0,
                                     (1 << specify_slots))

    result = homo_ops.homo_add_pt(cipher, encoded, cryptoContext)
    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    ground_truth = np.array(values) + np.array(values)
    if np.allclose(clear_result, ground_truth):
        print("homo_add_pt Test passed!")
    else:
        print("homo_add_pt Test failed!")
        print("result", clear_result[:len(values)])
        print("data", ground_truth)


    result = homo_ops.homo_mul_pt(cipher, encoded, cryptoContext)
    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    ground_truth = np.array(values) * np.array(values)
    if np.allclose(clear_result,ground_truth):
        print("homo_mul_pt Test passed!")
    else:
        print("homo_mul_pt Test failed!")
        print("result", clear_result[:len(values)])
        print("data", ground_truth)

    result = homo_ops.homo_add_pt(cipher, encoded, cryptoContext)
    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    ground_truth = np.array(values) + np.array(values)
    if np.allclose(clear_result, ground_truth):
        print("homo_add_pt second Test passed!")
    else:
        print("homo_add_pt second Test failed!")
        print("result", clear_result[:len(values)])
        print("data", ground_truth)