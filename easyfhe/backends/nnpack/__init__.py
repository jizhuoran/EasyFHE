# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

from easyfhe.backends import ContextProp, PropModule


__all__ = ["is_available", "flags", "set_flags"]


def is_available():
    return False


def _get_enabled():
    return False


def _set_enabled(enabled):
    if enabled:
        raise RuntimeError("NNPACK is disabled in EasyFHE")


def set_flags(_enabled):
    orig_flags = (_get_enabled(),)
    _set_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(enabled=False):
    orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        set_flags(orig_flags[0])


class NNPACKModule(PropModule):
    enabled = ContextProp(_get_enabled, _set_enabled)


sys.modules[__name__] = NNPACKModule(sys.modules[__name__], __name__)
