from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AOTConfig:
    fw_compiler: Any = None
    bw_compiler: Any = None
    partition_fn: Any = None
    decompositions: Any = None
    num_params_buffers: int = 0
    aot_id: int = 0
    keep_inference_input_mutations: bool = False
    dynamic_shapes: bool = False
    aot_autograd_arg_pos_to_source: Any = None
    is_export: bool = False
    no_tangents: bool = False
    enable_log: bool = False


def _disabled(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("AOTAutograd is disabled in EasyFHE")


def create_joint(*args: Any, **kwargs: Any) -> None:
    _disabled()


def from_fun(*args: Any, **kwargs: Any) -> None:
    _disabled()
