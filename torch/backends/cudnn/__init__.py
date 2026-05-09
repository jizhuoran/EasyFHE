# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

import torch
from torch.backends import _FP32Precision, ContextProp, PropModule

from . import rnn


def _disabled_if_true(value, name="cuDNN"):
    if value:
        raise RuntimeError(f"{name} is disabled in EasyFHE")


def version():
    return None


def is_available():
    return False


def is_acceptable(tensor):
    return False


def set_flags(
    _enabled=None,
    _benchmark=None,
    _benchmark_limit=None,
    _deterministic=None,
    _allow_tf32=None,
    _fp32_precision="none",
    _depthwise_kernel=None,
):
    orig_flags = (False, False, None, False, False, "none", "auto")
    if _enabled is not None:
        _disabled_if_true(_enabled)
    if _benchmark is not None:
        _disabled_if_true(_benchmark)
    return orig_flags


@contextmanager
def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
    fp32_precision="none",
    depthwise_kernel="auto",
):
    orig_flags = set_flags(
        enabled,
        benchmark,
        benchmark_limit,
        deterministic,
        allow_tf32,
        fp32_precision,
        depthwise_kernel,
    )
    try:
        yield
    finally:
        set_flags(*orig_flags)


class CudnnModule(PropModule):
    enabled = ContextProp(lambda: False, lambda val: _disabled_if_true(val))
    deterministic = ContextProp(lambda: False, lambda val: None)
    benchmark = ContextProp(lambda: False, lambda val: _disabled_if_true(val))
    benchmark_limit = None
    allow_tf32 = ContextProp(lambda: False, lambda val: None)
    conv = _FP32Precision("cuda", "conv")
    fp32_precision = ContextProp(lambda: "none", lambda val: None)
    depthwise_kernel = ContextProp(lambda: "auto", lambda val: None)


sys.modules[__name__] = CudnnModule(sys.modules[__name__], __name__)

enabled: bool
deterministic: bool
benchmark: bool
allow_tf32: bool
fp32_precision: str
benchmark_limit: int | None
depthwise_kernel: str
