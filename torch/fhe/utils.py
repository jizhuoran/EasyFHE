import time, os, pickle
from .client import client as client
from .client.gen_context import gen_contexts
from .context import *

# Global dictionary to accumulate execution time for each function
execution_times = {}

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
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/zrji/log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as profiler:
            result = func(*args, **kwargs)
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