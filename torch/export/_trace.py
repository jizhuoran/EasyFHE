from contextlib import contextmanager


@contextmanager
def custom_triton_ops_decomposition_disabled():
    yield
