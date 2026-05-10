from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def _set_node_metadata_hook(*args: Any, **kwargs: Any) -> Iterator[None]:
    yield
