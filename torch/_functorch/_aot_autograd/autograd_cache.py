from typing import Any


def autograd_cache_hash(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("AOTAutograd cache is disabled in EasyFHE")


def autograd_cache_key(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("AOTAutograd cache is disabled in EasyFHE")
