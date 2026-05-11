from __future__ import annotations

from typing import Any


class detect_anomaly:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("autograd anomaly detection is disabled in EasyFHE")


class set_detect_anomaly:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("autograd anomaly detection is disabled in EasyFHE")
