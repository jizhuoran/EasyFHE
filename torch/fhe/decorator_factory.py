from .compiler.compiler import frontend
from .debug_tool import auto_sync, check_meta_equal, plaintext_twin



def decorator_factory(func):
    decorators = [frontend, plaintext_twin, auto_sync]
    for dec in reversed(decorators):
        func = dec(func)
    return func