from contextlib import contextmanager

deterministic = False
force_cudagraph_gc = False


class _TritonConfig:
    cudagraphs = False


triton = _TritonConfig()


def get_config_copy():
    return {}


def get_type(name):
    raise AttributeError(name)


@contextmanager
def patch(*args, **kwargs):
    yield
