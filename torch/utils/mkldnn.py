# mypy: allow-untyped-defs
"""MKLDNN module conversion utilities are disabled in EasyFHE."""

from __future__ import annotations

import torch


def to_mkldnn(module, dtype=torch.float):
    raise RuntimeError("torch.utils.mkldnn is disabled in EasyFHE")
