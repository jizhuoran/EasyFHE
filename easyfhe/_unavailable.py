"""Helpers for PyTorch compatibility surfaces that EasyFHE intentionally omits."""

from __future__ import annotations


def feature_unavailable(feature: str, *, suggestion: str | None = None) -> None:
    message = f"{feature} is not available in EasyFHE"
    if suggestion:
        message = f"{message}; {suggestion}"
    raise RuntimeError(message)


class DisabledNamespace:
    def __init__(self, feature: str, *, suggestion: str | None = None) -> None:
        self._feature = feature
        self._suggestion = suggestion

    def __getattr__(self, name: str):
        feature_unavailable(f"{self._feature}.{name}", suggestion=self._suggestion)
