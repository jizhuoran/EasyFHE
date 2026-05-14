import importlib

import easyfhe as torch
from easyfhe._unavailable import feature_unavailable

from easyfhe.nn.parameter import (
    Buffer as Buffer,
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
from easyfhe.nn.modules.module import Module as Module


class _DisabledNNBindings:
    def __getattr__(self, name):
        raise AttributeError(f"easyfhe._C._nn.{name} is disabled in EasyFHE")


if not hasattr(torch._C, "_nn"):
    torch._C._nn = _DisabledNNBindings()


class _DisabledNNModule(Module):
    def __init__(self, *args, **kwargs):
        feature_unavailable(
            f"easyfhe.nn.{type(self).__name__}",
            suggestion="EasyFHE keeps tensor runtime pieces but omits training layers",
        )


_DISABLED_MODULE_NAMES = {
    "AdaptiveAvgPool2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "Linear",
    "ReLU",
    "Sequential",
}


for _name in _DISABLED_MODULE_NAMES:
    globals()[_name] = type(_name, (_DisabledNNModule,), {})


def __getattr__(name):
    if name in {"functional", "init", "modules"}:
        module = importlib.import_module(f"easyfhe.nn.{name}")
        globals()[name] = module
        return module
    if name in {
        "Parameter",
        "UninitializedParameter",
        "Buffer",
        "UninitializedBuffer",
        "Module",
        *_DISABLED_MODULE_NAMES,
    }:
        return globals()[name]
    raise AttributeError(f"easyfhe.nn.{name} is disabled in EasyFHE")


def factory_kwargs(kwargs):
    if kwargs is None:
        return {}
    simple_keys = {"device", "dtype", "memory_format"}
    expected_keys = simple_keys | {"factory_kwargs"}
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")
    result = dict(kwargs.get("factory_kwargs", {}))
    for key in simple_keys:
        if key in kwargs:
            if key in result:
                raise TypeError(f"{key} specified twice, in **kwargs and in factory_kwargs")
            result[key] = kwargs[key]
    return result


__all__ = [
    "Buffer",
    "Module",
    "Parameter",
    "UninitializedBuffer",
    "UninitializedParameter",
    "factory_kwargs",
    *_DISABLED_MODULE_NAMES,
]
