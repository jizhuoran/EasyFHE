# mypy: allow-untyped-defs
import contextlib

from torch._C import _SDPBackend as SDPBackend


__all__ = [
    "SDPBackend",
    "WARN_FOR_UNFUSED_KERNELS",
    "sdpa_kernel",
]


WARN_FOR_UNFUSED_KERNELS = False
SDPBackend.__module__ = __name__
SDPBackend.__name__ = "SDPBackend"
_current_backends = [SDPBackend.MATH]


def _raise_kernel_warnings(params) -> None:
    return None


def _backend_from_string(name: str):
    return getattr(SDPBackend, name)


def _cur_sdpa_kernel_backends(with_priority: bool = False):
    return list(_current_backends)


def _sdpa_kernel(backends, set_priority: bool = False) -> None:
    global _current_backends
    if isinstance(backends, SDPBackend):
        backends = [backends]
    _current_backends = list(dict.fromkeys(backends))


@contextlib.contextmanager
def sdpa_kernel(backends, set_priority: bool = False):
    previous_backends = _cur_sdpa_kernel_backends(with_priority=set_priority)
    _sdpa_kernel(backends, set_priority)
    try:
        yield None
    finally:
        _sdpa_kernel(previous_backends, set_priority)


def __getattr__(name):
    raise AttributeError(f"torch.nn.attention.{name} is disabled in EasyFHE")
