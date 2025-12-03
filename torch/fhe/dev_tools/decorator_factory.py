from .debug_tool import *
from .encode_tool import save_middle_encode, save_end_encode
from ..utils import call_counter, profile_python_function

def decorator_factory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cryptoContext = args[-1]
        if func.__name__ == "homo_bootstrap":
            cryptoContext.inBS = True
        if cryptoContext.inBS:
            result = func(*args, **kwargs)
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
            if config.CHECK_CIPHER:
                decorators.append(check_meta_equal)
            if config.SAVE_MIDDLE:
                decorators.append(save_middle_encode)
            if config.SAVE_END:
                decorators.append(save_end_encode)
            for dec in decorators:
                func1 = dec(func1)
            result = func1(*args, **kwargs)
        if func.__name__ == "homo_bootstrap":
            cryptoContext.inBS = False
        return result

    return wrapper
