import sys, os, time
import numpy as np

sys.path.append("/".join(os.getcwd().split("/")[:-3]))
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
import torch
import torch.fhe.bootstrapping as BS
import torch.fhe.utils as utils

DATA_DIR = os.environ["DATA_DIR"]
LOG_DIR = os.environ["LOG_DIR"]

maxLevelsRemaining=1
logBsSlots_list=[13]
logN=14
dnum=1
dcrtBits=52
firstMod=56
levelBudget_list=[[3,3]]
rescaleTech = "FLEXIBLEAUTO"  # "FLEXIBLEAUTO" # "FIXEDMANUAL" # "FIXEDAUTO"
path = DATA_DIR
secretKeyDist = "UNIFORM_TERNARY"  # "SPARSE_TERNARY"  "UNIFORM_TERNARY"

config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True, COMPARE_WITH_OPENFHE=False)

cryptoContext, openfhe_context = utils.try_load_context(
    int(maxLevelsRemaining),
    [],
    logBsSlots_list,
    int(logN),
    int(dnum),
    int(dcrtBits),
    int(firstMod),
    levelBudget_list,
    secretKeyDist,
    rescaleTech,
    save_dir=path,
    config=config,
)

logBsSlots = logBsSlots_list[0]

# Test the correctness of the bootstrapping
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
x = np.array([values[i % len(values)] for i in range((1 << logBsSlots))])
x = torch.tensor(x, device="cuda")
cipher = openfhe_context.encrypt(
    x, 1, openfhe_context.depth - 1, (1 << logBsSlots)
)  # specify the slots value explicitly

cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]
cryptoContext.BsContext.to_cuda()

# keyset1 = list()
# for key in cryptoContext.slots_left_rot_key_map[str(logBsSlots_list[0])]:
#     keyset1.append(key)
# keyset2 = set()
# for key in cryptoContext.slots_left_rot_key_map[str(logBsSlots_list[1])]:
#     keyset2.add(key)
# keyset3 = set()
# for key in cryptoContext.slots_left_rot_key_map[str(logBsSlots_list[2])]:
#     keyset3.add(key)

# exit(0)

# print("num of keys in keyset1:", len(keyset1))
# print("L", cryptoContext.L)
# print("num of keys in keyset2:", len(keyset2))
# print("num of keys in keyset3:", len(keyset3))
# print("num of smae keys:", len(keyset1.intersection(keyset2.intersection(keyset3))))


cryptoContext.load_rotation_keys(logBsSlots)

# with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(
#             LOG_DIR
#         ),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#     ) as profiler:
#         # Start profiling specific functions with torch.profiler.record_function()
#         result = BS.eval_bootstrap(cipher, cryptoContext.L, logBsSlots=logBsSlots, cryptoContext=cryptoContext)
#         profiler.step()

# # Get the profiling results
# profiler_results = profiler.key_averages()

# # Print the profiling summary in a table format
# print(profiler_results.table(sort_by="self_cuda_time_total"))


TEST_COMPILE = False

if TEST_COMPILE:
    result1 = BS.eval_bootstrap(
        cipher,
        cryptoContext.L,
        logBsSlots_list[0],
        cryptoContext,
    )
    start_time = time.time()
    result1 = BS.eval_bootstrap(
        cipher,
        cryptoContext.L,
        logBsSlots_list[0],
        cryptoContext,
    )
    print("Time taken for NORMAL bootstrapping:", time.time() - start_time)
    print("=======================")
    print("=======================")
    print("=======================")
    result2 = COMPILE.eval_bootstrap(
        cipher,
        cryptoContext.L,
        logBsSlots_list[0],
        cryptoContext,
    )
    start_time = time.time()
    result2 = COMPILE.eval_bootstrap(
        cipher,
        cryptoContext.L,
        logBsSlots_list[0],
        cryptoContext,
    )
    print("Time taken for COMPILE bootstrapping:", time.time() - start_time)
    print("result1", result1.cv[0].cpu().numpy()[0][:10])
    print("result2", result2.cv[0].cpu().numpy()[0][:10])

    if np.array_equal(result1.cv[0].cpu().numpy(), result2.cv[0].cpu().numpy()):
        print("Test passed!")
        print("Test passed!")
        print("Test passed!")
    else:
        print("Test failed!")
        print("Test failed!")
        print("Test failed!")



    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                DATA_DIR+"/log"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            # Start profiling specific functions with torch.profiler.record_function()
            result = BS.eval_bootstrap(cipher, cryptoContext.L, logBsSlots, cryptoContext)
            profiler.step()

    # Get the profiling results
    profiler_results = profiler.key_averages()

    # Print the profiling summary in a table format
    print(profiler_results.table(sort_by="self_cpu_time_total"))

    print("++++++++")
    print("++++++++")
    print("++++++++")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                DATA_DIR+"/log"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            # Start profiling specific functions with torch.profiler.record_function()
            result = COMPILE.eval_bootstrap(cipher, cryptoContext.L, logBsSlots, cryptoContext)
            profiler.step()

    # Get the profiling results
    profiler_results = profiler.key_averages()

    # Print the profiling summary in a table format
    print(profiler_results.table(sort_by="self_cpu_time_total"))


else:

    # result = BS.eval_bootstrap(
    #     cipher, cryptoContext.L, logBsSlots, cryptoContext
    # )

    start_time = time.time()
    result = BS.eval_bootstrap(
        cipher, cryptoContext.L, logBsSlots, cryptoContext
    )
    print("Time taken for bootstrapping:", time.time() - start_time)
    # openfhe_boot_context = openfhe_boot_contexts[str(logBsSlots)]
    # openfhe_result = openfhe_boot_context.cc.EvalBootstrap(cipher_openfhe)
    # data = np.array(openfhe_result.GetVectorOfData(), dtype=np.uint64)

    clear_result = openfhe_context.decrypt(result)  # decrypt by cc with different slots value should be fine
    clear_result = clear_result.cpu().numpy().reshape(-1)
    print("HE decryption result: ", clear_result[:10])

    # is_equal = utils.compare_gpufhe_ct_with_openfhe(result, openfhe_result)
    # if is_equal:
    #     print("Test passed!")
    # else:
    #     print("Test failed!")
    #     print("result", result.cv[0].cpu().numpy()[0][:10])
    #     print("data", data.reshape(-1)[:10])
