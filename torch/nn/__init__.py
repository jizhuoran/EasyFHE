# mypy: allow-untyped-defs
from torch.nn.parameter import (  # usort: skip
    Buffer as Buffer,
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
import torch
import importlib


if not hasattr(torch._C, "_nn"):
    class _DisabledNNBindings:
        pass

    torch._C._nn = _DisabledNNBindings()


from torch.nn import parameter as parameter


def __getattr__(name):
    if name in {"functional", "init", "modules"}:
        module = importlib.import_module(f"torch.nn.{name}")
        globals()[name] = module
        return module
    if name == "Module":
        from torch.nn.modules.module import Module
        globals()[name] = Module
        return Module
    if name in {"ModuleDict", "ModuleList", "Sequential"}:
        module = importlib.import_module("torch.nn.modules")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "Conv2d":
        from torch.nn.modules.conv import Conv2d
        globals()[name] = Conv2d
        return Conv2d
    if name in {"BatchNorm1d", "BatchNorm2d"}:
        module = importlib.import_module("torch.nn.modules")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "Linear":
        from torch.nn.modules.linear import Linear
        globals()[name] = Linear
        return Linear
    if name == "AdaptiveAvgPool2d":
        from torch.nn.modules.pooling import AdaptiveAvgPool2d
        globals()[name] = AdaptiveAvgPool2d
        return AdaptiveAvgPool2d
    raise AttributeError(f"torch.nn.{name} is disabled in EasyFHE fast build")


def factory_kwargs(kwargs):
    r"""Return a canonicalized dict of factory kwargs.

    Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
    to factory functions like torch.empty, or errors if unrecognized kwargs are present.

    This function makes it simple to write code like this::

        class MyModule(nn.Module):
            def __init__(self, **kwargs):
                factory_kwargs = torch.nn.factory_kwargs(kwargs)
                self.weight = Parameter(torch.empty(10, **factory_kwargs))

    Why should you use this function instead of just passing `kwargs` along directly?

    1. This function does error validation, so if there are unexpected kwargs we will
    immediately report an error, instead of deferring it to the factory call
    2. This function supports a special `factory_kwargs` argument, which can be used to
    explicitly specify a kwarg to be used for factory functions, in the event one of the
    factory kwargs conflicts with an already existing argument in the signature (e.g.
    in the signature ``def f(dtype, **kwargs)``, you can specify ``dtype`` for factory
    functions, as distinct from the dtype argument, by saying
    ``f(dtype1, factory_kwargs={"dtype": dtype2})``)
    """
    if kwargs is None:
        return {}
    simple_keys = {"device", "dtype", "memory_format"}
    expected_keys = simple_keys | {"factory_kwargs"}
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")

    # guarantee no input kwargs is untouched
    r = dict(kwargs.get("factory_kwargs", {}))
    for k in simple_keys:
        if k in kwargs:
            if k in r:
                raise TypeError(
                    f"{k} specified twice, in **kwargs and in factory_kwargs"
                )
            r[k] = kwargs[k]

    return r
