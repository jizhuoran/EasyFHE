# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from torch.backends import ContextProp, PropModule


def is_available():
    return False


def is_acl_available():
    return False


def _get_enabled():
    return False


def _set_enabled(enabled):
    if enabled:
        raise RuntimeError("MKLDNN is disabled in EasyFHE")


VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2


class _PrecisionModule:
    fp32_precision = "none"


class verbose:
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level != VERBOSE_OFF:
            raise RuntimeError("MKLDNN is disabled in EasyFHE")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def set_flags(
    _enabled=None, _deterministic=None, _allow_tf32=None, _fp32_precision="none"
):
    orig_flags = (_get_enabled(), False, False, "none")
    if _enabled is not None:
        _set_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(enabled=False, deterministic=False, allow_tf32=True, fp32_precision="none"):
    orig_flags = set_flags(enabled, deterministic, allow_tf32, fp32_precision)
    try:
        yield
    finally:
        set_flags(*orig_flags)


class MkldnnModule(PropModule):
    def is_available(self):
        return is_available()

    enabled = ContextProp(_get_enabled, _set_enabled)
    deterministic = ContextProp(lambda: False, lambda val: None)
    allow_tf32 = ContextProp(lambda: False, lambda val: None)
    matmul = _PrecisionModule()
    conv = _PrecisionModule()
    rnn = _PrecisionModule()
    fp32_precision = ContextProp(lambda: "none", lambda val: None)


if TYPE_CHECKING:
    enabled: ContextProp
    deterministic: ContextProp
    allow_tf32: ContextProp
    fp32_precision: str


sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)
