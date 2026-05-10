from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _add_batch_dim(x: Any, *args: Any, **kwargs: Any) -> Any:
    return x


def _broadcast_to_and_flatten(x: Any, *args: Any, **kwargs: Any) -> Any:
    return x


def restore_vmap(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("functorch vmap is disabled in EasyFHE")


def unwrap_batched(x: Any, *args: Any, **kwargs: Any) -> Any:
    return x, None


def wrap_batched(x: Any, *args: Any, **kwargs: Any) -> Any:
    return x


def lazy_load_decompositions() -> None:
    return None


@contextmanager
def doesnt_support_saved_tensors_hooks(*args: Any, **kwargs: Any) -> Iterator[None]:
    yield
