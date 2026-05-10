"""Tiny EasyFHE stub for the disabled torch.compile stack."""


def disable(fn=None, recursive=True, wrapping=True):
    if fn is None:
        return lambda inner: inner
    return fn


def graph_break(*args, **kwargs):
    return None


def optimize(*args, **kwargs):
    raise RuntimeError("torch.compile is disabled in EasyFHE")


def reset():
    return None


def list_backends(*args, **kwargs):
    return []

