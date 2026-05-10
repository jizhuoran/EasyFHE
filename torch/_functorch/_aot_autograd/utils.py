from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

KNOWN_TYPES: set[type[Any]] = set()


def simple_wraps(fn: Callable[..., Any]) -> Callable[[_F], _F]:
    def wrap(inner: _F) -> _F:
        return wraps(fn)(inner)

    return wrap


def top_saved_tensors_hooks(*args: Any, **kwargs: Any) -> tuple[None, None]:
    return (None, None)


def saved_tensors_hooks_are_inlineable(*args: Any, **kwargs: Any) -> bool:
    return False


def is_with_effects(*args: Any, **kwargs: Any) -> bool:
    return False
