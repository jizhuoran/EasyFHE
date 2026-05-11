"""EasyFHE stubs for the disabled torch.compile stack."""

from contextlib import contextmanager


def _disabled(*args, **kwargs):
    raise RuntimeError("torch.compile is disabled in EasyFHE")


def compile(*args, **kwargs):
    return _disabled(*args, **kwargs)


def reset():
    return None


def allow_in_graph(fn):
    return fn


def assume_constant_result(fn):
    return fn


def substitute_in_graph(original_fn, **kwargs):
    def decorator(fn):
        return fn

    return decorator


def disable(fn=None, **kwargs):
    if fn is None:
        return lambda inner: inner
    return fn


def list_backends(*args, **kwargs):
    return []


def set_default_backend(*args, **kwargs):
    return None


def get_default_backend():
    return "eager"


@contextmanager
def set_stance(*args, **kwargs):
    yield


def set_enable_guard_collectives(*args, **kwargs):
    return None


def cudagraph_mark_step_begin():
    return None


def load_compiled_function(*args, **kwargs):
    return _disabled(*args, **kwargs)


def wrap_numpy(fn=None):
    if fn is None:
        return lambda inner: inner
    return fn


def is_compiling():
    return False


def is_dynamo_compiling():
    return False


def is_exporting():
    return False


def save_cache_artifacts(*args, **kwargs):
    return None


def load_cache_artifacts(*args, **kwargs):
    return None


def _identity_guard_filter(*args, **kwargs):
    return True


keep_portable_guards_unsafe = _identity_guard_filter
keep_global_context_and_tensor_guards_unsafe = _identity_guard_filter
skip_guard_on_inbuilt_nn_modules_unsafe = _identity_guard_filter
skip_guard_on_all_nn_modules_unsafe = _identity_guard_filter
keep_tensor_guards_unsafe = _identity_guard_filter
keep_tensor_guards = _identity_guard_filter
skip_guard_on_globals_unsafe = _identity_guard_filter
skip_guard_on_globals = _identity_guard_filter
skip_all_guards_unsafe = _identity_guard_filter


@contextmanager
def nested_compile_region(*args, **kwargs):
    yield
