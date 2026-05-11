from __future__ import annotations

from typing import Any


def gradcheck(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("torch.autograd.gradcheck is disabled in EasyFHE")


def gradgradcheck(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("torch.autograd.gradgradcheck is disabled in EasyFHE")
