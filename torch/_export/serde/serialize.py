from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SerializedArtifact:
    data: bytes = b""


def serialize(*args: Any, **kwargs: Any) -> SerializedArtifact:
    raise RuntimeError("torch._export serialization is disabled in EasyFHE")


def deserialize(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("torch._export serialization is disabled in EasyFHE")


def _bytes_to_dataclass(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("torch._export serialization is disabled in EasyFHE")
