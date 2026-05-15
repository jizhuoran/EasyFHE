"""Minimal native anchor for sparse tensor classes.

EasyFHE does not expose the PyTorch sparse Python API, but the native tensor
extension still imports this module while installing sparse tensor classes.
"""

__all__: list[str] = []
