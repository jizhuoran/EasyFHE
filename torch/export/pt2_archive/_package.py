from typing import Any

AOTI_FILES = "aoti_files"


def package_pt2(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.export is disabled in EasyFHE")


def load_pt2(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.export is disabled in EasyFHE")


def is_pt2_package(*args: Any, **kwargs: Any) -> bool:
    return False
