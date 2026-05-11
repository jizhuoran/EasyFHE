from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class InfoProtocol(Protocol):
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


@dataclass
class Info:
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


def make_autograd_impl(op, info: InfoProtocol) -> Callable:
    def autograd_impl(*args, **kwargs):
        raise RuntimeError("custom operator autograd is disabled in EasyFHE")

    return autograd_impl
