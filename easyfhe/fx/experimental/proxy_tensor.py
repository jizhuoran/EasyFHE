try:
    from easyfhe.utils._python_dispatch import TorchDispatchMode
except Exception:
    class TorchDispatchMode:
        pass


class ProxyTorchDispatchMode(TorchDispatchMode):
    pass


class PreDispatchTorchFunctionMode:
    pass


class PythonKeyTracer:
    pass


_CURRENT_MAKE_FX_TRACER = None
_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT = {}


def get_proxy_mode():
    return None


def make_fx(*args, **kwargs):
    raise RuntimeError("torch.fx is disabled in EasyFHE")


def track_tensor_tree(output, proxy, constant=None, tracer=None):
    return output


def wrapper_and_args_for_make_fx(fn, args, kwargs=None):
    return fn, args
