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

def call_counter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1  # Increment the call count
        return func(*args, **kwargs)
    
    wrapper.count = 0  # Initialize the call count
    call_registry[func.__name__] = wrapper  # Register the function
    return wrapper


@atexit.register
def print_call_counts():
    print("\nFunction Call Counts:")
    for func_name, wrapper in call_registry.items():
        print(f"Function '{func_name}' was called {wrapper.count} times.")

@atexit.register
def print_execution_times():
    print("\nExecution Times:")
    for func_name, exec_time in execution_times.items():
        print(f"Function '{func_name}' executed in {exec_time:.6f} seconds.")

def check_meta_equal(func):
    def wrapper(*args, **kwargs):
        in0, in1 = args[0], args[1]
        # assert len(in0.cv) == len(in1.cv)
        # assert in0.cur_limbs == in1.cur_limbs
        # assert in0.scaling_factor == in1.scaling_factor
        # assert in0.noise_deg == in1.noise_deg
        # assert in0.is_ext == in1.is_ext
        # assert in0.slots == in1.slots
        return func(*args, **kwargs)
    return wrapper

def check_cipher_len(func):
    def wrapper(*args, **kwargs):
        assert len(args[0].cv) == 2
        return func(*args, **kwargs)
    return wrapper


def profile_python_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculate the execution time for this call
        exec_time = end_time - start_time

        # Update the global dictionary with the accumulated time for this function
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = 0
        execution_times[func.__name__] += exec_time

        # print(f"Function {func.__name__} executed in {exec_time:.6f} seconds")
        return result

    return wrapper


def profile_pytorch_function(func):
    def wrapper(*args, **kwargs):
        # Set up the profiler
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/zrji/log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as profiler:
            result = func(*args, **kwargs)
            profiler.step()

        profiler_results = profiler.key_averages()
        print(profiler_results.table(sort_by="self_cuda_time_total"))
        print(profiler_results.table(sort_by="self_cpu_time_total"))

        return result

    return wrapper

def round_half_away_from_zero(number, ndigits=0):
    multiplier = 10 ** ndigits
    if number > 0:
        return math.floor(number * multiplier + 0.5) / multiplier
    elif number < 0:
        return math.ceil(number * multiplier - 0.5) / multiplier
    else:
        return 0.0

def try_load_context(logN,
            logSlots,
            maxLevelsRemaining,
            levelBudget,
            dnum,
            dcrtBits,
            firstMod,
            approxModDepth,
            secretKeyDist,
            rescaleTech,
            save_dir):

    load_path = (
        save_dir
        + "/GPU-FHE-CONTEXT_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
            logN,
            logSlots,
            maxLevelsRemaining,
            levelBudget[0],
            levelBudget[1],
            dnum,
            dcrtBits,
            firstMod,
            approxModDepth,
            secretKeyDist,
            rescaleTech,
        )
    )

    if not os.path.exists(load_path):
        gen_contexts(
            logN=logN,
            logSlots=logSlots, # possible slots value of runtime ciphertext #todo: should be a list?
            maxLevelsRemaining=maxLevelsRemaining,
            levelBudget=levelBudget,
            dnum=dnum,
            dcrtBits=dcrtBits,
            firstMod=firstMod,
            approxModDepth=approxModDepth,
            rotate_index=[],
            secretKeyDist="UNIFORM_TERNARY",
            rescaleTech=rescaleTech,
            save_dir=save_dir
        )

    with open(load_path, 'rb') as file:
        gpufheMembers, openfheMembers, BsContextMembers = pickle.load(file)


    openfhe_context = client.OpenFHEContext(openfheMembers)
    cryptoContext = Context(BsContextMembers, gpufheMembers)

    return cryptoContext, openfhe_context

def compare_bs_ct_with_openfhe(bs_cipher, openfhe_cipher):
    gpu_bootstrapping_res = np.array([bs_cipher.cv[0].cpu().numpy(), bs_cipher.cv[1].cpu().numpy()]).reshape(-1)
    openfhe_bootstrapping_res = np.array(openfhe_cipher.GetVectorOfData()).reshape(-1)
    return np.array_equal(gpu_bootstrapping_res, openfhe_bootstrapping_res)
