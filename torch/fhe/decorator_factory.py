from .compiler.compiler import frontend
from .debug_tool import auto_sync, check_meta_equal



def decorator_factory(func):
    decorators = [auto_sync, frontend]
    for dec in reversed(decorators):
        func = dec(func)
    return func