from dataclasses import dataclass


@dataclass(frozen=True)
class DVar:
    name: str | None = None
