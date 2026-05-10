"""APIs related to torch.compile."""

import functools
from collections.abc import Callable
from typing import overload, TypeVar
from typing_extensions import ParamSpec


_T = TypeVar("_T")
_P = ParamSpec("_P")


@overload
def _disable_dynamo(
    fn: Callable[_P, _T], recursive: bool = True
) -> Callable[_P, _T]: ...


@overload
def _disable_dynamo(
    fn: None = None, recursive: bool = True
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def _disable_dynamo(
    fn: Callable[_P, _T] | None = None, recursive: bool = True
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.
    """
    if fn is not None:
        return fn
    else:
        # decorator usage like @_disable_dynamo(recursive=False). The resulting
        # object expects the original decorated function as the arg.
        return functools.partial(_disable_dynamo, recursive=recursive)
