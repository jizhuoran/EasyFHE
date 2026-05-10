from typing import Any


def runtime_wrapper(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("AOTAutograd runtime wrappers are disabled in EasyFHE")


def aot_dispatch_autograd(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("AOTAutograd runtime wrappers are disabled in EasyFHE")
