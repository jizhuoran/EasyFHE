from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def custom_triton_ops_decomposition_disabled() -> Iterator[None]:
    yield


def is_exporting() -> bool:
    return False


def _export(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.export is disabled in EasyFHE")


def _export_to_torch_ir(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.export is disabled in EasyFHE")
