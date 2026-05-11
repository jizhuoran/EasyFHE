from __future__ import annotations

from typing import Any


def _disabled(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("torch.autograd.functional is disabled in EasyFHE")


jacobian = _disabled
hessian = _disabled
vjp = _disabled
jvp = _disabled
vhp = _disabled
hvp = _disabled
