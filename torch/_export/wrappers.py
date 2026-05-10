from __future__ import annotations

from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


def mark_subclass_constructor_exportable_experimental(fn: _T) -> _T:
    return fn


def allow_in_pre_dispatch_graph(fn: _T) -> _T:
    return fn


def mark_constructor_exportable_experimental(fn: _T) -> _T:
    return fn
