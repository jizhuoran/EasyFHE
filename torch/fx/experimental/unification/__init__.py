from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Var:
    name: str | None = None


def unify(lhs: Any, rhs: Any, *args: Any, **kwargs: Any) -> dict[Any, Any]:
    return {}
