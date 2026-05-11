from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def _disabled(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("forward-mode autograd is disabled in EasyFHE")


@contextmanager
def dual_level():
    _disabled()
    yield


enter_dual_level = _disabled
exit_dual_level = _disabled
make_dual = _disabled
unpack_dual = lambda tensor, *args, **kwargs: (tensor, None)
