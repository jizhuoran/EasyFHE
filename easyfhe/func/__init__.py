from __future__ import annotations

from typing import Any


def _disabled(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.func is disabled in EasyFHE")


functional_call = _disabled
functionalize = _disabled
grad = _disabled
grad_and_value = _disabled
hessian = _disabled
jacfwd = _disabled
jacrev = _disabled
jvp = _disabled
linearize = _disabled
replace_all_batch_norm_modules_ = _disabled
stack_module_state = _disabled
vjp = _disabled
vmap = _disabled

__all__ = [
    "functional_call",
    "functionalize",
    "grad",
    "grad_and_value",
    "hessian",
    "jacfwd",
    "jacrev",
    "jvp",
    "linearize",
    "replace_all_batch_norm_modules_",
    "stack_module_state",
    "vjp",
    "vmap",
]
