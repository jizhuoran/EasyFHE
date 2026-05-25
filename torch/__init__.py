"""Compatibility entry point for packages that import ``torch``.

EasyFHE exposes the PyTorch-compatible runtime as ``easyfhe``.  Some external
tools, including Triton, still import a small set of symbols from ``torch``
directly.  Keep this shim intentionally thin and forward those imports to the
EasyFHE runtime.
"""

import sys as _sys

import easyfhe as _easyfhe

for _name in dir(_easyfhe):
    if _name not in {
        "__builtins__",
        "__cached__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
    }:
        globals()[_name] = getattr(_easyfhe, _name)

_C = _easyfhe._C
cuda = _easyfhe.cuda
version = _easyfhe.version

_sys.modules.setdefault("torch", _sys.modules[__name__])
_sys.modules["torch._C"] = _C
_sys.modules["torch.cuda"] = cuda
_sys.modules["torch.version"] = version

__all__ = getattr(_easyfhe, "__all__", [])

del _easyfhe, _name, _sys
