from __future__ import annotations

from typing import Any, Callable, TypeVar


_F = TypeVar("_F", bound=Callable[..., Any])


def _disabled() -> None:
    raise RuntimeError("torch.autograd.Function is disabled in EasyFHE")


class FunctionCtx:
    def save_for_backward(self, *tensors: Any) -> None:
        self.saved_tensors = tensors

    def save_for_forward(self, *tensors: Any) -> None:
        self.saved_for_forward = tensors

    def mark_dirty(self, *args: Any) -> None:
        _disabled()

    def mark_non_differentiable(self, *args: Any) -> None:
        _disabled()

    def set_materialize_grads(self, value: bool) -> None:
        self.materialize_grads = value


class BackwardCFunction:
    pass


class FunctionMeta(type):
    pass


class Function(metaclass=FunctionMeta):
    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        _disabled()


class InplaceFunction(Function):
    pass


class NestedIOFunction(Function):
    pass


def once_differentiable(fn: _F) -> _F:
    return fn


def custom_function_call(*args: Any, **kwargs: Any) -> Any:
    _disabled()
