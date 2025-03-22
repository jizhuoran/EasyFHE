from datetime import datetime
import time, os, pickle
import numpy as np
import functools
import atexit
from .client import client as client
from .client.gen_context import gen_contexts
from .context import *
import torch

# Global dictionary to accumulate execution time for each function
execution_times = {}

# Registry to keep track of function call counts
call_registry = {}

@atexit.register
def print_call_counts():
    if len(call_registry) > 0:
        print("\nFunction Call Counts:")
        for func_name, wrapper in call_registry.items():
            print(f"Function '{func_name}' was called {wrapper.count} times.")


@atexit.register
def print_execution_times():
    if len(execution_times) > 0:
        total_time = sum(execution_times.values())
        print("\nExecution Times:")
        for func_name, exec_time in execution_times.items():
            print(f"Function '{func_name}' executed in {exec_time:.6f} seconds.")
        print(f"Total execution time profiled: {total_time:.6f} seconds.")

def call_counter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1  # Increment the call count
        return func(*args, **kwargs)
    wrapper.count = 0  # Initialize the call count
    call_registry[func.__name__] = wrapper  # Register the function
    return wrapper

def profile_python_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cpu.synchronize()
        torch.cuda.synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cpu.synchronize()
        torch.cuda.synchronize()
        end_time = time.time()
        # Calculate the execution time for this call
        exec_time = end_time - start_time
        # Update the global dictionary with the accumulated time for this function
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = 0
        execution_times[func.__name__] += exec_time
        return result
    return wrapper


def profile_pytorch_function(func):
    def wrapper(*args, **kwargs):
        # Set up the profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/zrji/log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            result = func(*args, **kwargs)
            profiler.step()

        profiler_results = profiler.key_averages()
        print(profiler_results.table(sort_by="self_cuda_time_total"))
        print(profiler_results.table(sort_by="self_cpu_time_total"))

        return result

    return wrapper


def round_half_away_from_zero(number, ndigits=0):
    multiplier = 10**ndigits
    if number > 0:
        return math.floor(number * multiplier + 0.5) / multiplier
    elif number < 0:
        return math.ceil(number * multiplier - 0.5) / multiplier
    else:
        return 0.0


def try_load_context(
    maxLevelsRemaining,
    rotIndex_list,
    logBsSlots_list,
    logN,
    dnum,
    dcrtBits,
    firstMod,
    levelBudget_list,
    secretKeyDist,
    rescaleTech,
    save_dir,
    config
):

    NO_BS=False
    if logBsSlots_list is None or logBsSlots_list == []:
        assert (logBsSlots_list is None or logBsSlots_list == []) == (levelBudget_list is None or levelBudget_list == []), \
            "ERROR: logBsSlots_list and levelBudget_list must be both None or both not None!"
        logBsSlots_list = [0]
        levelBudget_list = [[0, 0]]
        NO_BS = True
    else:
        sorted_pairs = sorted(
            zip(logBsSlots_list, levelBudget_list), key=lambda x: x[0]
        )
        logBsSlots_list, levelBudget_list = zip(*sorted_pairs)
        logBsSlots_list = list(logBsSlots_list)
        levelBudget_list = list(levelBudget_list)

    load_path = save_dir + "/GPU-FHE-CONTEXT_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
        maxLevelsRemaining,
        "-".join(map(str, logBsSlots_list)),
        "-".join("-".join(map(str, levelBudget)) for levelBudget in levelBudget_list),
        logN,
        dnum,
        dcrtBits,
        firstMod,
        secretKeyDist,
        rescaleTech,
    )

    debug_load_path = (
        save_dir
        + "/DEBUG-GPU-FHE-CONTEXT_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            maxLevelsRemaining,
            "-".join(map(str, logBsSlots_list)),
            "-".join(
                "-".join(map(str, levelBudget)) for levelBudget in levelBudget_list
            ),
            logN,
            dnum,
            dcrtBits,
            firstMod,
            secretKeyDist,
            rescaleTech,
        )
    )

    if (not os.path.exists(load_path)) or (
        not os.path.exists(debug_load_path) and config.COMPARE_WITH_OPENFHE == "debug"
    ):
        gen_contexts(
            maxLevelsRemaining=maxLevelsRemaining,
            rotIndex_list=rotIndex_list,
            logBsSlots_list=logBsSlots_list,
            logN=logN,
            dnum=dnum,
            dcrtBits=dcrtBits,
            firstMod=firstMod,
            levelBudget_list=levelBudget_list,
            secretKeyDist=secretKeyDist,
            rescaleTech=rescaleTech,
            save_dir=save_dir,
            config=config,
        )

    with open(load_path, "rb") as file:
        gpufheMembers, openfheMembers, BsContextMembers = pickle.load(file)

    if config.COMPARE_WITH_OPENFHE:
        if not os.path.exists(debug_load_path):
            print("ERROR: There is no debug context file! Please regenerate context!")
        with open(debug_load_path, "rb") as file:
            debug_keys = pickle.load(file)

    cryptoContext = Context(BsContextMembers, gpufheMembers, config)
    if cryptoContext.config.AUTO_LOAD_KEYS:
        if rotIndex_list is not None and rotIndex_list != []:
            cryptoContext.load_rotation_keys("app")
        if NO_BS == False:
            for logBsSlots in logBsSlots_list:
                cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]
                cryptoContext.BsContext.to_cuda()
                cryptoContext.load_rotation_keys(logBsSlots)

    openfhe_context = client.OpenFHEContext(openfheMembers)
    openfhe_context.config = cryptoContext.config
    cryptoContext.openfhe_context = openfhe_context
    if config.COMPARE_WITH_OPENFHE:
        openfhe_boot_contexts = {}
        if NO_BS == False:
            for logBsSlots, level_budget in zip(logBsSlots_list, levelBudget_list):
                openfhe_boot_contexts[str(logBsSlots)] = client.OpenFHEContext(
                    openfheMembers
                )
                openfhe_boot_contexts[str(logBsSlots)].setup_for_debug(
                    debug_keys, 1 << logBsSlots, level_budget
                )
                openfhe_boot_contexts[str(logBsSlots)].config = cryptoContext.config
        
        return cryptoContext, openfhe_context, openfhe_boot_contexts
    else:
        return cryptoContext, openfhe_context


def compare_bs_ct_with_openfhe(bs_cipher, openfhe_cipher):
    gpu_bootstrapping_res = np.array(
        [bs_cipher.cv[0][:bs_cipher.cur_limbs].cpu().numpy(), bs_cipher.cv[1][:bs_cipher.cur_limbs].cpu().numpy()]
    ).reshape(-1)
    openfhe_bootstrapping_res = np.array(openfhe_cipher.GetVectorOfData()).reshape(-1)
    return np.array_equal(gpu_bootstrapping_res, openfhe_bootstrapping_res)

