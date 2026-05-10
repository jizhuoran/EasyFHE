# mypy: allow-untyped-defs
from contextlib import contextmanager

from torch.futures import Future


def _disabled(*args, **kwargs):
    raise RuntimeError("torch.jit is disabled in EasyFHE fast build")


def is_scripting():
    return False


def is_tracing():
    return False


@contextmanager
def optimized_execution(should_optimize):
    yield


script = _disabled
script_method = _disabled
trace = _disabled
trace_module = _disabled
load = _disabled
save = _disabled
freeze = _disabled
fork = _disabled
wait = _disabled

__all__ = [
    "freeze",
    "fork",
    "Future",
    "is_scripting",
    "is_tracing",
    "load",
    "optimized_execution",
    "save",
    "script",
    "script_method",
    "trace",
    "trace_module",
    "wait",
]
