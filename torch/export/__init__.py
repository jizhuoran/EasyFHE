from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dynamic_shapes import (
    AdditionalInputs,
    Constraint,
    Dim,
    dims,
    ShapesCollection,
)
from .exported_program import (
    default_decompositions,
    ExportedProgram,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
from .unflatten import FlatArgsAdapter, unflatten, UnflattenedModule

CustomDecompTable = dict


def _disabled(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.export is disabled in EasyFHE")


def export(*args: Any, **kwargs: Any) -> None:
    _disabled()


def draft_export(*args: Any, **kwargs: Any) -> None:
    _disabled()


def save(*args: Any, **kwargs: Any) -> None:
    _disabled()


def load(*args: Any, **kwargs: Any) -> None:
    _disabled()


def register_dataclass(cls: type[Any], *args: Any, **kwargs: Any) -> type[Any]:
    return cls


__all__ = [
    "AdditionalInputs",
    "Constraint",
    "CustomDecompTable",
    "Dim",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "FlatArgsAdapter",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "ShapesCollection",
    "UnflattenedModule",
    "default_decompositions",
    "dims",
    "draft_export",
    "export",
    "load",
    "register_dataclass",
    "save",
    "unflatten",
]
