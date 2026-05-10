from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

enable_auto_functionalized_v2_for_export = False
detect_non_strict_fake_tensor_leaks = False
use_new_tracer_experimental = False


@contextmanager
def patch(**kwargs: Any) -> Iterator[None]:
    old_values = {name: globals().get(name) for name in kwargs}
    globals().update(kwargs)
    try:
        yield
    finally:
        globals().update(old_values)
