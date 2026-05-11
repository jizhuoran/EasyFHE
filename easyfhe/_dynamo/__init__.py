"""Tiny EasyFHE stub for the disabled torch.compile stack."""


def _disabled(*args, **kwargs):
    raise RuntimeError("torch.compile is disabled in EasyFHE")


def disable(fn=None, recursive=True, wrapping=True):
    if fn is None:
        return lambda inner: inner
    return fn


def allow_in_graph(fn):
    return fn


def assume_constant_result(fn):
    return fn


def graph_break(*args, **kwargs):
    return None


def is_compiling():
    return False


def is_dynamo_compiling():
    return False


def mark_dynamic(*args, **kwargs):
    return None


def mark_static(*args, **kwargs):
    return None


def optimize(*args, **kwargs):
    return _disabled(*args, **kwargs)


def reset():
    return None


def list_backends(*args, **kwargs):
    return []


def substitute_in_graph(original_fn, **kwargs):
    def decorator(fn):
        return fn

    return decorator
