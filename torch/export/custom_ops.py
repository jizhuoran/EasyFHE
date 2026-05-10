from typing import Any


def _call_custom_autograd_function_in_pre_dispatch(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.export is disabled in EasyFHE")
