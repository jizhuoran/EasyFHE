from dataclasses import dataclass

registered_hop_fake_fns = {}


def _disabled(*args, **kwargs):
    raise RuntimeError("higher-order ops are disabled in EasyFHE")


def _in_hop_compile():
    return False


def has_user_subclass(*args, **kwargs):
    return False


def _has_gen_schema(*args, **kwargs):
    return False


def autograd_not_implemented(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


def _tensor_storage(tensor):
    return tensor.untyped_storage()


def reenter_make_fx(*args, **kwargs):
    return _disabled(*args, **kwargs)


def materialize_as_graph(*args, **kwargs):
    return _disabled(*args, **kwargs)


def _maybe_run_with_interpreter(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _maybe_compile_and_run_fn(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _hop_compile_and_call(*args, **kwargs):
    return _disabled(*args, **kwargs)


def first_slice_copy(value):
    return value[0].clone()


@dataclass
class HopInstance:
    op: object
    operands: tuple


class FunctionalizeCtxWrapper:
    pass


class SubgraphCallableWrapper:
    def __init__(self, subgraph):
        self.subgraph = subgraph

    def __call__(self, *args, **kwargs):
        return self.subgraph(*args, **kwargs)
