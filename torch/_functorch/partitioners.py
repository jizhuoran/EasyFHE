from __future__ import annotations

from enum import Enum
from typing import Any


class CheckpointPolicy(Enum):
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3


def min_cut_rematerialization_partition(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("AOTAutograd partitioners are disabled in EasyFHE")
