import time, pickle, math
import numpy as np
import functools
import atexit
from .context import *
import easyfhe as torch

# Global dictionary to accumulate execution time for each function
execution_times = {}

# Registry to keep track of function call counts
call_registry = {}


def _openfhe_client_removed():
    raise RuntimeError(
        "The legacy OpenFHE client/context path has been removed from EasyFHE. "
        "Use easyfhe.fhe.generate_context(...) for native EasyFHE context generation."
    )


class _EasyFHEUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch" or module.startswith("torch."):
            module = "easyfhe" + module[len("torch") :]
        return super().find_class(module, name)


def load_pickle(file):
    return _EasyFHEUnpickler(file).load()

@atexit.register
def print_call_counts():
    if len(call_registry) > 0:
        print("\nFunction Call Counts:")
        for func_name, count in call_registry.items():
            print(f"Function '{func_name}' was called {count} times.")


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
        if func.__name__ not in call_registry:
            call_registry[func.__name__] = 1
        else:
            call_registry[func.__name__] += 1
        return func(*args, **kwargs)
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
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("~"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            result = func(*args, **kwargs)
            profiler.step()

        profiler_results = profiler.key_averages()
        if torch.cuda.is_available():
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
    device,
    save_dir,
    config
):
    _openfhe_client_removed()


def compare_gpufhe_ct_with_openfhe(bs_cipher, openfhe_cipher):
    gpu_bootstrapping_res = np.array(
        [bs_cipher.cv[0][:bs_cipher.cur_limbs].cpu().numpy(), bs_cipher.cv[1][:bs_cipher.cur_limbs].cpu().numpy()]
    ).reshape(-1)
    openfhe_bootstrapping_res = np.array(openfhe_cipher.GetVectorOfData()).reshape(-1)
    return np.array_equal(gpu_bootstrapping_res, openfhe_bootstrapping_res)

def compare_cpufhe_with_gpufhe(cpufhe_ct, gpufhe_ct):
    # cpu_res = np.array([cpufhe_ct.cv[0].cpu().numpy()]).reshape(-1)
    # gpu_res = np.array([gpufhe_ct.cv[0].cpu().numpy()]).reshape(-1)
    cpu_res = np.array([cpufhe_ct.cv[0].cpu().numpy(), cpufhe_ct.cv[1].cpu().numpy()]).reshape(-1)
    gpu_res = np.array([gpufhe_ct.cv[0].cpu().numpy(), gpufhe_ct.cv[1].cpu().numpy()]).reshape(-1)
    return np.array_equal(cpu_res, gpu_res)
