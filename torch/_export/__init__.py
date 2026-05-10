from __future__ import annotations

from typing import Any


def _disabled(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch._export is disabled in EasyFHE")


def aot_compile(*args: Any, **kwargs: Any) -> None:
    _disabled()


def aot_load(*args: Any, **kwargs: Any) -> None:
    _disabled()


__all__ = ["aot_compile", "aot_load"]
