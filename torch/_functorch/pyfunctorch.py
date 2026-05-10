import torch

TransformType = torch._C._functorch.TransformType


def dispatch_functorch(*args, **kwargs):
    raise RuntimeError("functorch is disabled in EasyFHE")


def retrieve_current_functorch_interpreter():
    return None


class temporarily_clear_interpreter_stack:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class temporarily_restore_interpreter_stack:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
