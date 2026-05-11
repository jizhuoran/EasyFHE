# mypy: allow-untyped-defs
import sys
import types


def is_available():
    return False


class _XNNPACKEnabled:
    def __get__(self, obj, objtype):
        return False

    def __set__(self, obj, val):
        if val:
            raise RuntimeError("XNNPACK is disabled in EasyFHE")


class XNNPACKEngine(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    enabled = _XNNPACKEnabled()


sys.modules[__name__] = XNNPACKEngine(sys.modules[__name__], __name__)
