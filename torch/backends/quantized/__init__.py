# mypy: allow-untyped-defs
import sys
import types


class _QEngineProp:
    def __get__(self, obj, objtype) -> str:
        return "none"

    def __set__(self, obj, val: str) -> None:
        if val not in ("none", "", None):
            raise RuntimeError("Quantized engines are disabled in EasyFHE")


class _SupportedQEnginesProp:
    def __get__(self, obj, objtype) -> list[str]:
        return ["none"]

    def __set__(self, obj, val) -> None:
        raise RuntimeError("Assignment not supported")


class QuantizedEngine(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    engine = _QEngineProp()
    supported_engines = _SupportedQEnginesProp()


sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)
engine: str
supported_engines: list[str]
