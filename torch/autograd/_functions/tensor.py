from __future__ import annotations

from typing import Any


class Resize:
    @staticmethod
    def apply(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("autograd Resize is disabled in EasyFHE")
