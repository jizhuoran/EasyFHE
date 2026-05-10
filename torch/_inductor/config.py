from contextlib import contextmanager


def get_config_copy():
    return {}


def get_type(name):
    raise AttributeError(name)


@contextmanager
def patch(*args, **kwargs):
    yield

