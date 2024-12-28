import time
from .client import client as client
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


def save_context(cryptoContext, openfhe_context, path = "torch/fhe/data/"):
    with open(path + 'crypto.pkl', 'wb') as file:
        pickle.dump((cryptoContext.Serialize(), openfhe_context.Serialize()), file)


def load_context(path = "torch/fhe/data/"):
    with open(path + 'crypto.pkl', 'rb') as file:
        cryptoContext_byte, openfhe_context_byte = pickle.load(file)
    openfhe_context = client.OpenFHEContext.Deserialize(openfhe_context_byte)
    cryptoContext = Context.Deserialize(cryptoContext_byte)
    return cryptoContext, openfhe_context