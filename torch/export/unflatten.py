from typing import Any


class FlatArgsAdapter:
    pass


class UnflattenedModule:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("torch.export is disabled in EasyFHE")


def unflatten(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.export is disabled in EasyFHE")
