import functools
from .compiler.compiler import frontend
from .debug_tool import *
from .utils import call_counter, profile_python_function

def decorator_factory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cryptoContext = args[-1]
        if func.__name__ == "homo_bootstrap":
            cryptoContext.inBS = True
        if cryptoContext.inBS:
            return func(*args, **kwargs)
        else: # not in BS
            config = cryptoContext.config
            func1 = func
            decorators = []
            if config.COUNT_OPS:
                decorators.append(call_counter)
            if config.TIME_OPS:
                decorators.append(profile_python_function)
            if config.AUTO_SYNC:
                decorators.append(auto_sync)
            if config.PTX_TWIN:
                decorators.append(plaintext_twin)
            if config.COMPILER:
                decorators.append(frontend)
            if config.CHECK_CIPHER:
                decorators.append(check_meta_equal)
            for dec in decorators:
                func1 = dec(func1)
        result = func1(*args, **kwargs)
        if func.__name__ == "homo_bootstrap":
            cryptoContext.inBS = False
        return result

    return wrapper