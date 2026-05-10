from __future__ import annotations

from typing import Any, Callable


def vmap(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]:
    def wrapped(*inner_args: Any, **inner_kwargs: Any) -> Any:
        raise RuntimeError("functorch transforms are disabled in EasyFHE")

    return wrapped
