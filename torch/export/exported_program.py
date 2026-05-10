from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ExportedProgram:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("torch.export is disabled in EasyFHE")


@dataclass
class ModuleCallEntry:
    fqn: str = ""


@dataclass
class ModuleCallSignature:
    inputs: tuple[Any, ...] = ()
    outputs: tuple[Any, ...] = ()


def default_decompositions() -> dict[Any, Any]:
    return {}


class ConstantArgument:
    pass


class TensorArgument:
    pass


class ModuleCallSignatureEntry:
    pass
