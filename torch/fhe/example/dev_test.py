from ..client.gen_context import gen_contexts
from ..bs_context import *
from .. import homo_ops
from .. import utils
from ..bootstrapping import eval_bootstrap
import numpy as np
import os

def BootstrapTest_N65536L26lB44(
    logN=14,
    logSlots_list=[13],
    maxLevelsRemaining=3,
    levelBudget_list=[[4, 4]],
    dnum=3,
    dcrtBits=59,
    firstMod=60,
    approxModDepth=9,
    rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
    save_dir="torch/fhe/data/"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    force_update_context = False
    # Force update the context
    if force_update_context:
        gen_contexts(
                logN=logN,
                logSlots_list=logSlots_list, # possible slots value of runtime ciphertext #todo: should be a list?
                maxLevelsRemaining=maxLevelsRemaining,
                levelBudget_list=levelBudget_list,
                dnum=dnum,
                dcrtBits=dcrtBits,
                firstMod=firstMod,
                approxModDepth=approxModDepth,
                rotate_index=[],
                secretKeyDist="UNIFORM_TERNARY",
                rescaleTech=rescaleTech,
                save_dir=save_dir
            )

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
            save_dir=save_dir)

    openfhe_context = openfhe_context_dict[str(logSlots_list[0])]
    dim1 = [0, 0]

    # Test the correctness of the bootstrapping
    logSlots = logSlots_list[0]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<logSlots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<logSlots)

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(logSlots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, logSlots)
    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)
    openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)

    is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
    if is_euqal:
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")

    else:
        print("BootstrapTest_N65536L26lB44: Test failed!")
        print("BootstrapTest_N65536L26lB44: Test failed!")
        print("BootstrapTest_N65536L26lB44: Test failed!")

    exit()

    measure_execution_time = True
    if measure_execution_time:
        start = time.time()
        result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)
        end = time.time()
        print("time", end - start)

        # Print the accumulated execution times
        # print("\nTotal execution time for each function:")
        # sorted_execution_times = sorted(utils.execution_times.items(), key=lambda x: x[1], reverse=True)
        # for func_name, total_time in sorted_execution_times:
        #     print(f"{func_name}: {total_time:.6f} seconds")

        pytorch_profiling = False
        if pytorch_profiling:
            # Set up the profiler
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "/home/zrji/log"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:
                # Start profiling specific functions with torch.profiler.record_function()
                result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots,
                                        cryptoContext=cryptoContext)

            # Get the profiling results
            profiler_results = profiler.key_averages()

            # Print the profiling summary in a table format
            print(profiler_results.table(sort_by="self_cpu_time_total"))


def BootstrapTest_slots_list_example(
        logN=14,
        logSlots_list=[11, 12],
        maxLevelsRemaining=3,
        levelBudget_list=[[3, 3], [4, 4]],
        dnum=3,
        dcrtBits=59,
        firstMod=60,
        approxModDepth=9,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="torch/fhe/data/"

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
                                                            save_dir=save_dir)

    dim1 = [0, 0]

    # logslots = 11
    specify_slots = logSlots_list[0]
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<specify_slots)

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)

    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    #test correctness
    openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
    is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
    if is_euqal:
        print("BootstrapTest_logslots11: Test passed!")
    else:
        print("BootstrapTest_logslots11: Test failed!")

    # logslots = 12
    specify_slots = logSlots_list[1]
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<specify_slots)

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)

    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    #test correctness
    openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
    is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
    if is_euqal:
        print("BootstrapTest_logslots12: Test passed!")
    else:
        print("BootstrapTest_logslots12: Test failed!")


def BootstrapTest_test_case(
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
    print("gpu bootstrapp done!")
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

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)
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


def Keyswitch_test_case(
        logN=14,
        logSlots_list=[11],
        maxLevelsRemaining=2,
        levelBudget_list=[[4, 4]],
        dnum=3,
        dcrtBits=50,
        firstMod=54,
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
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, 0, 1<<specify_slots)

    # do the application computation
    utils.load_rotation_keys(cryptoContext, "app")
    print("L ", cryptoContext.L)
    print("K ", cryptoContext.K)
    print("alpha ", cryptoContext.alpha)
    cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    cipher = homo_ops.homo_square(cipher, cryptoContext)
    cipher = homo_ops.homo_square(cipher, cryptoContext)
    cipher = homo_ops.homo_square(cipher, cryptoContext)
    cipher = homo_ops.homo_square(cipher, cryptoContext)
    cipher = homo_ops.homo_square(cipher, cryptoContext)
    cipher = homo_ops.homo_square(cipher, cryptoContext)
    print("2rot, 6sqr")
    print("gpu bootstrapp done!")
    # compute golden answer
    if mode == "debug":
        cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe, -1)
        cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe, 2)
        cipher_openfhe = openfhe_context.cc.EvalSquare(cipher_openfhe)
        cipher_openfhe = openfhe_context.cc.EvalSquare(cipher_openfhe)
        cipher_openfhe = openfhe_context.cc.EvalSquare(cipher_openfhe)
        cipher_openfhe = openfhe_context.cc.EvalSquare(cipher_openfhe)
        cipher_openfhe = openfhe_context.cc.EvalSquare(cipher_openfhe)
        cipher_openfhe = openfhe_context.cc.EvalSquare(cipher_openfhe)
        is_euqal = utils.compare_bs_ct_with_openfhe(cipher, cipher_openfhe)
        if is_euqal:
            print("Key switch: Test passed!")
        else:
            print("Key switch: Test failed!")

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
                                                                 [-1,2],
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