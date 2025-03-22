from .compiler.compiler import frontend
from .debug_tool import auto_sync, check_meta_equal



def decorator_factory(func):
    decorators = [frontend, auto_sync]
    for dec in reversed(decorators):
        func = dec(func)
    return func