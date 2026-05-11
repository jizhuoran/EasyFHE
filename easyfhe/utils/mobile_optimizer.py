# mypy: allow-untyped-defs
"""Mobile model optimization is disabled in EasyFHE."""

from __future__ import annotations

from enum import Enum
from typing import AnyStr

import easyfhe as torch


class MobileOptimizerType(Enum):
    CONV_BN_FUSION = 0
    INSERT_FOLD_PREPACK_OPS = 1
    REMOVE_DROPOUT = 2
    FUSE_ADD_RELU = 3
    HOIST_CONV_PACKED_PARAMS = 4
    VULKAN_AUTOMATIC_GPU_TRANSFER = 5


class LintCode(Enum):
    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4


def optimize_for_mobile(
    script_module: torch.jit.ScriptModule,
    optimization_blocklist: set[MobileOptimizerType] | None = None,
    preserved_methods: list[AnyStr] | None = None,
    backend: str = "CPU",
) -> torch.jit.RecursiveScriptModule:
    raise RuntimeError("torch.utils.mobile_optimizer is disabled in EasyFHE")


def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    return []
