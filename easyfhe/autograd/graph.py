from __future__ import annotations

from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Callable


GradientEdge = namedtuple("GradientEdge", ["node", "output_nr"])


class Node:
    def name(self) -> str:
        return "EasyFHEDisabledAutogradNode"

    @property
    def next_functions(self) -> tuple:
        return ()

    def metadata(self) -> dict[str, Any]:
        return {}

    def register_hook(self, fn: Callable[..., Any]) -> Any:
        raise RuntimeError("autograd hooks are disabled in EasyFHE")

    def register_prehook(self, fn: Callable[..., Any]) -> Any:
        raise RuntimeError("autograd hooks are disabled in EasyFHE")


def get_gradient_edge(tensor: Any) -> GradientEdge:
    raise RuntimeError("autograd graph edges are disabled in EasyFHE")


def increment_version(tensor: Any) -> None:
    return None


def register_multi_grad_hook(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("autograd hooks are disabled in EasyFHE")


def set_warn_on_accumulate_grad_stream_mismatch(enabled: bool) -> None:
    return None


def set_override_stale_capture_stream(enabled: bool) -> None:
    return None


@contextmanager
def saved_tensors_hooks(*args: Any, **kwargs: Any):
    yield


@contextmanager
def save_on_cpu(*args: Any, **kwargs: Any):
    yield


@contextmanager
def disable_saved_tensors_hooks(*args: Any, **kwargs: Any):
    yield


@contextmanager
def allow_mutation_on_saved_tensors(*args: Any, **kwargs: Any):
    yield
