import sys, os, warnings
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch.fhe.homo_ops as homo_ops
from torch.fhe.bootstrapping import eval_bootstrap, homo_double_bootstrap, homo_bootstrap
import torch.fhe.utils as utils
import torch.fhe.bs_context
import numpy as np
from termcolor import colored

DATA_DIR = os.environ["DATA_DIR"]

def print_failed(message):
    print(colored(message, "red"))

def app_without_bs_example_debug(
        maxLevelsRemaining=5,
        appRotIndex_list = [-1, 2, -4, 5],
        logBsSlots_list=None,
        logN=14,
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        levelBudget_list=None,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir=DATA_DIR
):

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=False, COMPARE_WITH_OPENFHE=True)
    cryptoContext, openfhe_context, _ = (
        utils.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir,
                               config=config))

    encode_slots = (1 << 11)
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

    # do the application computation
    cryptoContext.load_rotation_keys("app")
    cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, -4, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 5, cryptoContext)
    print("homo_rotate done!")
    # compute golden answer
    cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe, -1)
    cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe,2)
    cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe,-4)
    cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe,5)
    is_euqal = utils.compare_gpufhe_ct_with_openfhe(cipher, cipher_openfhe)
    if is_euqal:
        print("homo_rotate: Test passed!")
    else:
        print_failed("homo_rotate: Test failed!")

def app_example_debug(
        maxLevelsRemaining=3,
        appRotIndex_list = [-1, 2],
        logBsSlots_list=[12, 13],
        logN=14,
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        levelBudget_list=[[3, 3], [4, 4]],
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir=DATA_DIR
):

    config = torch.fhe.config.Config(CHECK_CIPHER=False, PTX_TWIN=False, AUTO_LOAD_KEYS=False, COMPARE_WITH_OPENFHE=True) #eval_bootstrap and PTX_TWIN cannot pass CHECK_CIPHER
    cryptoContext, openfhe_context, openfhe_boot_contexts = (
        utils.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir,
                               config=config))

    encode_slots = (1 << 11)
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

    # do the application computation
    cryptoContext.load_rotation_keys("app")
    cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    print("homo_rotate done!")
    # compute golden answer

    cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe, -1)
    cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe,2)
    is_euqal = utils.compare_gpufhe_ct_with_openfhe(cipher, cipher_openfhe)
    if is_euqal:
        print("homo_rotate: Test passed!")
    else:
        print_failed("homo_rotate: Test failed!")

    # bootstrapping
    cryptoContext.load_bootstrapping_context(str(logBsSlots_list[0]))
    result = eval_bootstrap(cipher, cryptoContext.L, logBsSlots_list[0], cryptoContext)
    result = homo_ops.homo_rescale(result, 1, cryptoContext)
    print("gpu bootstrapp done!")
    # compute golden answer

    cipher_openfhe.SetSlots((1<<logBsSlots_list[0]))
    openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots_list[0])]
    openfhe_boot = openfhe_boot_context.cc.EvalBootstrap(cipher_openfhe)
    openfhe_boot = openfhe_context.cc.ModReduce(openfhe_boot)
    is_euqal = utils.compare_gpufhe_ct_with_openfhe(result, openfhe_boot)
    if is_euqal:
        print("BootstrapTest_logBsSlots11: Test passed!")
    else:
        print_failed("BootstrapTest_logBsSlots11: Test failed!")

    # do some multiplication to consume some limbs
    result.slots =  (1 << 12) # This assignment is for testing purposes only
    drop_limbs = result.cur_limbs - 3
    for i in range(drop_limbs):
        result = homo_ops.homo_mul(result, result, cryptoContext)
        result = homo_ops.homo_rescale(result, 1, cryptoContext)

    # bootstrapping
    cryptoContext.load_bootstrapping_context(str(logBsSlots_list[1]))
    result1 = eval_bootstrap(result, cryptoContext.L, logBsSlots_list[1], cryptoContext)
    result1 = homo_ops.homo_rescale(result1, 1, cryptoContext)
    print("gpu bootstrapp done!")

    # do some multiplication to consume some limbs
    for i in range(drop_limbs):
        openfhe_boot = openfhe_context.cc.EvalSquare(openfhe_boot)
        openfhe_boot = openfhe_context.cc.ModReduce(openfhe_boot)

    openfhe_boot.SetSlots((1 << logBsSlots_list[1])) # to cheat openfhe boot with bs_slots = (1<<logBsSlots_list[1])
    openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots_list[1])]
    openfhe_boot1 = openfhe_boot_context.cc.EvalBootstrap(openfhe_boot)
    openfhe_boot1 = openfhe_context.cc.ModReduce(openfhe_boot1)

    is_euqal = utils.compare_gpufhe_ct_with_openfhe(result1, openfhe_boot1)
    if is_euqal:
        print("BootstrapTest_logBsSlots12: Test passed!")
    else:
        print_failed("BootstrapTest_logBsSlots12: Test failed!")


def app_example_release(
        maxLevelsRemaining=3,
        appRotIndex_list = [-1, 2],
        logBsSlots_list=[11, 12],
        logN=14,
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        levelBudget_list=[[3, 3], [4, 4]],
        rescaleTech="FLEXIBLEAUTO",  # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir=DATA_DIR,
        AUTO_LOAD_KEYS=True
):

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=AUTO_LOAD_KEYS)
    cryptoContext, openfhe_context = (
        utils.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir,
                               config=config))

    print("Current allocated memory (GB):", torch.cuda.memory_allocated() / 1024 / 1024 / 1024)

    encode_slots = (1 << 11)
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    x = torch.tensor(x, device="cuda")
    cipher = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

    values1 = [0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888, 0.888888]
    x1 = np.array([values1[i % len(values1)] for i in range(encode_slots)])
    x1 = torch.tensor(x1, device="cuda")
    cipher1 = openfhe_context.encrypt(x1, 1, 0, encode_slots)

    # do the application computation
    cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    print("homo_rotate done!")

    # bootstrapping
    result = homo_bootstrap(cipher, cryptoContext.L, logBsSlots_list[0], cryptoContext)
    print("gpu bootstrapp done!")

    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[:10])

    # some homomorphic computation
    result.slots = (1 << 12)  # This assignment is for testing purposes only.
    approx_plain_val = clear_result[:10] # compute approx golden answer
    # print(approx_plain_val)
    for i in range(result.cur_limbs - 4):
        approx_plain_val = approx_plain_val * values1[0]
        # print(approx_plain_val)
        result = homo_ops.homo_mul(result, cipher1, cryptoContext)
        result = homo_ops.homo_rescale(result, 1, cryptoContext)

    # bootstrapping
    result1 = homo_bootstrap(result, cryptoContext.L, logBsSlots_list[1], cryptoContext)
    print("gpu bootstrapp done!")

    clear_result = openfhe_context.decrypt(result1)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    warnings.warn("note: openfhe adds random noise during decryption, therefore the result might be slightly different each time, "
          "and might be different from the openfhe decryption result even in the same round")
    print("plain result: ", approx_plain_val)
    print("HE decryption result: ", clear_result[:10])

    is_equal = np.allclose(clear_result[:10], approx_plain_val[:10], atol=1e-02)
    if is_equal:
        print("app: Test passed!")
    else:
        print_failed("app: Test failed! The code verifies if the first 10 elements of clear_result and approx_plain_val are approximately equal, allowing a maximum difference of 0.01 (1e-2). Please review the results.")


def encode_test_case(
        maxLevelsRemaining=6,
        logBsSlots_list=None,
        logN=14,
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        levelBudget_list=None,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir=DATA_DIR
):
    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=False, COMPARE_WITH_OPENFHE=True, SAVE_MIDDLE=True)
    cryptoContext, openfhe_context, _ = (
        utils.try_load_context(maxLevelsRemaining, [], logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir,
                               config=config))
    ############
    ## test 1 ##
    ############
    x = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])
    encode_slots = (1<<10)
    plaintext = homo_ops.encode(x, "test1", 0, encode_slots, cryptoContext)
    plaintext_golden = openfhe_context.encode(x, 1, 0, encode_slots)

    all_correct = True
    attributes = [
        ('slots', plaintext.slots, plaintext_golden.slots),
        ('noise_deg', plaintext.noise_deg, plaintext_golden.noise_deg),
        ('scaling_factor', plaintext.scaling_factor, plaintext_golden.scaling_factor),
        ('cur_limbs', plaintext.cur_limbs, plaintext_golden.cur_limbs),
        ('len', len(plaintext.cv), len(plaintext_golden.cv)),
    ]

    # Compare attributes
    for attr_name, attr_value, golden_value in attributes:
        if attr_value != golden_value:
            all_correct = False
            print(f"{attr_name}: {attr_value} != {golden_value}")

    # Compare cv values
    for i in range(len(plaintext.cv)):
        if not torch.equal(plaintext.cv[i], plaintext_golden.cv[i]):
            all_correct = False
            break

    if all_correct:
        print("encode with specify slots Test passed!")
    else:
        print_failed("encode with specify slots Test failed!")

    # ############
    # ## test 2 ##
    # ############
    # encode_slots = (1 << 11)
    # values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    # x = np.array([values[i % len(values)] for i in range(encode_slots)])
    # x = torch.tensor(x, device="cuda")
    # cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, 0, encode_slots)
    # encoded = homo_ops.encode(x, 0, encode_slots, cryptoContext)

    # result = homo_ops.homo_add_pt(cipher, encoded, cryptoContext)
    # clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    # clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    # ground_truth = np.array(values) + np.array(values)
    # if np.allclose(clear_result, ground_truth):
    #     print("homo_add_pt with gpu_fft Test passed!")
    # else:
    #     print_failed("homo_add_pt with gpu_fft Test failed!")
    #     print("result", clear_result[:len(values)])
    #     print("data", ground_truth)

    # ############
    # ## test 3 ##
    # ############
    # encode_slots = (1 << 11)
    # values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    # x = np.array(values)
    # x = torch.tensor(x, device="cuda")
    # cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, 0, encode_slots)
    # encoded = homo_ops.encode(x, 0, encode_slots, cryptoContext)

    # result = homo_ops.homo_add_pt(cipher, encoded, cryptoContext)
    # clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    # clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    # ground_truth = np.array(values) + np.array(values)
    # if np.allclose(clear_result, ground_truth):
    #     print("homo_add_pt with gpu_fft Test passed!")
    # else:
    #     print_failed("homo_add_pt with gpu_fft Test failed!")
    #     print("result", clear_result[:len(values)])
    #     print("data", ground_truth)


    ############
    ## test 4 ##
    ############
    cryptoContext.config.SAVE_MIDDLE = False
    x = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])
    encode_slots = (1<<10)
    pre_encode_value = homo_ops.pre_encode(x, encode_slots)
    pre_encode_value.encoded_values = torch.tensor(pre_encode_value.encoded_values, device="cuda", dtype=torch.double)
    plaintext = homo_ops.encode(pre_encode_value, "test4", 0, encode_slots, cryptoContext)

    plaintext_golden = openfhe_context.encode(x, 1, 0, encode_slots)

    all_correct = True
    attributes = [
        ('slots', plaintext.slots, plaintext_golden.slots),
        ('noise_deg', plaintext.noise_deg, plaintext_golden.noise_deg),
        ('scaling_factor', plaintext.scaling_factor, plaintext_golden.scaling_factor),
        ('cur_limbs', plaintext.cur_limbs, plaintext_golden.cur_limbs),
        ('len', len(plaintext.cv), len(plaintext_golden.cv)),
    ]

    # Compare attributes
    for attr_name, attr_value, golden_value in attributes:
        if attr_value != golden_value:
            all_correct = False
            print(f"{attr_name}: {attr_value} != {golden_value}")

    # Compare cv values
    for i in range(len(plaintext.cv)):
        if not torch.equal(plaintext.cv[i], plaintext_golden.cv[i]):
            all_correct = False
            break

    if all_correct:
        print("encode from middle Test passed!")
    else:
        print_failed("encode from middle Test failed!")


def ct_pt_test_case(
        maxLevelsRemaining=6,
        logBsSlots_list=None,
        logN=14,
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        levelBudget_list=None,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir=DATA_DIR,
        plaintext_twin = False
):

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=False, COMPARE_WITH_OPENFHE=True, PTX_TWIN = plaintext_twin)
    cryptoContext, openfhe_context, _ = (
        utils.try_load_context(maxLevelsRemaining, [], logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir,
                               config=config))

    encode_slots=(1 << 11)
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, 0, encode_slots)
    encoded = openfhe_context.encode(values, 1, 0, encode_slots)

    result = homo_ops.homo_add_pt(cipher, encoded, cryptoContext)
    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    if plaintext_twin:
        clear_result = np.array(result.ptx_twin).reshape(-1)[:len(values)]
    ground_truth = np.array(values) + np.array(values)
    if np.allclose(clear_result, ground_truth):
        print("homo_add_pt Test passed!")
    else:
        print_failed("homo_add_pt Test failed!")
        print("result", clear_result[:len(values)])
        print("data", ground_truth)


    result = homo_ops.homo_mul_pt(cipher, encoded, cryptoContext)
    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    ground_truth = np.array(values) * np.array(values)
    if plaintext_twin:
        clear_result = np.array(result.ptx_twin).reshape(-1)[:len(values)]
    if np.allclose(clear_result,ground_truth):
        print("homo_mul_pt Test passed!")
    else:
        print_failed("homo_mul_pt Test failed!")
        print("result", clear_result[:len(values)])
        print("data", ground_truth)

    # encoded = homo_ops._drop_last_elements(encoded, 1, cryptoContext, False)
    result = homo_ops.homo_rescale(result, 1, cryptoContext)
    encoded = homo_ops.adjust_to(encoded, result.cur_limbs, result.noise_deg, result.scaling_factor, cryptoContext)
    result = homo_ops.homo_add_pt(result, encoded, cryptoContext)
    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    ground_truth = np.array(ground_truth) + np.array(values)
    if plaintext_twin:
        clear_result = np.array(result.ptx_twin).reshape(-1)[:len(values)]
    if np.allclose(clear_result, ground_truth):
        print("homo_add_pt second Test passed!")
    else:
        print_failed("homo_add_pt second Test failed!")
        print("result", clear_result[:len(values)])
        print("data", ground_truth)

    result = homo_ops.homo_mul_pt(result, encoded, cryptoContext)
    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    ground_truth = np.array(ground_truth) * np.array(values)
    if plaintext_twin:
        clear_result = np.array(result.ptx_twin).reshape(-1)[:len(values)]
    if np.allclose(clear_result, ground_truth):
        print("homo_mul_pt second Test passed!")
    else:
        print_failed("homo_mul_pt second Test failed!")
        print("result", clear_result[:len(values)])
        print("data", ground_truth)


def double_bs_debug(
        maxLevelsRemaining=3,
        appRotIndex_list = [],
        logBsSlots_list=[11],
        logN=14,
        dnum=3,
        dcrtBits=52,
        firstMod=56,
        levelBudget_list=[[3, 3]],
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir=DATA_DIR,
        mode = "debug" # "debug" or "release"
):

    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, COMPARE_WITH_OPENFHE=True)
    cryptoContext, openfhe_context, openfhe_boot_contexts = (
        utils.try_load_context(maxLevelsRemaining, appRotIndex_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                               levelBudget_list, "UNIFORM_TERNARY", rescaleTech, save_dir=save_dir,
                               config=config))

    openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots_list[0])]
    encode_slots = (1 << 11)
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range(encode_slots)])
    x = torch.tensor(x, device="cuda")
    openfhe_boot_context.config = openfhe_context.config
    cipher, cipher_openfhe = openfhe_boot_context.encrypt(x, 1, openfhe_context.depth - 1, encode_slots)

    precision = 17

    # bootstrapping
    cryptoContext.load_bootstrapping_context(str(logBsSlots_list[0]))
    result = homo_double_bootstrap(cipher, L0=cryptoContext.L, logBsSlots=logBsSlots_list[0], precision=precision, cryptoContext=cryptoContext)
    print("gpu bootstrapp done!")
    clear_result = openfhe_boot_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)[:len(values)]
    print("clear result ", clear_result[:10])
    # compute golden answer
    if config.COMPARE_WITH_OPENFHE == True:
        openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots_list[0])]
        num_iter = 2
        openfhe_boot = openfhe_boot_context.cc.EvalBootstrap(cipher_openfhe, num_iter, precision)
        openfhe_boot = openfhe_boot_context.cc.ModReduce(openfhe_boot)
        is_euqal = utils.compare_gpufhe_ct_with_openfhe(result, openfhe_boot)
        if is_euqal:
            print("BootstrapTest_logBsSlots11: Test passed!")
        else:
            print_failed("BootstrapTest_logBsSlots11: Test failed!")

##############
## run tests #
##############

if __name__ == "__main__":
    for rescaleTech in ["FLEXIBLEAUTO", "FIXEDAUTO", "FIXEDMANUAL"]:
        print("***********{}***********".format(rescaleTech))
        print("==========={}============".format('app_without_bs_example_debug'))
        app_without_bs_example_debug(rescaleTech = rescaleTech)
        print("==========={}============".format('app_example_debug'))
        app_example_debug(rescaleTech = rescaleTech)
        print("==========={}============".format('app_example_release NOT AUTO_LOAD_KEYS'))
        app_example_release(rescaleTech = rescaleTech, AUTO_LOAD_KEYS=False)
        print("==========={}============".format('app_example_release AUTO_LOAD_KEYS'))
        app_example_release(rescaleTech = rescaleTech, AUTO_LOAD_KEYS=True)
        print("==========={}============".format('encode_test_case'))
        encode_test_case(rescaleTech = rescaleTech)
        print("==========={}============".format('ct_pt_test_case'))
        ct_pt_test_case(rescaleTech = rescaleTech, plaintext_twin = False)
        print("==========={}============".format('test_plaintext_twin'))
        ct_pt_test_case(rescaleTech = rescaleTech, plaintext_twin = True)
        print("==========={}============".format('double_bs_debug'))
        double_bs_debug(rescaleTech = rescaleTech)
        print("************************************".format(rescaleTech))
