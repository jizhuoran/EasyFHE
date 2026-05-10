from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InputKind(Enum):
    USER_INPUT = 0


class OutputKind(Enum):
    USER_OUTPUT = 0


@dataclass
class ExportBackwardSignature:
    pass


@dataclass
class ExportGraphSignature:
    pass


@dataclass
class CustomObjArgument:
    name: str = ""
