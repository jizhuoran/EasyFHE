from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Constraint:
    name: str = ""


class Dim:
    AUTO = object()
    DYNAMIC = object()
    STATIC = object()

    def __init__(self, name: str | None = None, *, min: int | None = None, max: int | None = None) -> None:
        self.__name__ = name or ""
        self.min = min
        self.max = max


class _DerivedDim(Dim):
    pass


class ShapesCollection(dict[Any, Any]):
    pass


class AdditionalInputs:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def dims(*names: str, min: int | None = None, max: int | None = None) -> tuple[Dim, ...]:
    return tuple(Dim(name, min=min, max=max) for name in names)


def _get_dim_name_mapping(dynamic_shapes: Any) -> dict[str, Any]:
    return {}


def refine_dynamic_shapes_from_suggested_fixes(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("torch.export is disabled in EasyFHE")


def _tree_map_with_path(fn: Any, tree: Any, *rests: Any, **kwargs: Any) -> Any:
    return tree


class _IntWrapper:
    def __init__(self, val: int) -> None:
        self.val = val
