from __future__ import annotations

from enum import Enum
from typing import Any

try:
    from torch._C._onnx import OperatorExportTypes, TensorProtoDataType, TrainingMode
except ImportError:

    class OperatorExportTypes(Enum):
        ONNX = 0

    class TensorProtoDataType(Enum):
        UNDEFINED = 0

    class TrainingMode(Enum):
        EVAL = 0


class OnnxExporterError(RuntimeError):
    pass


class ONNXProgram:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disabled()


class ExportableModule:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disabled()


class InputObserver:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disabled()


producer_name = "pytorch"
producer_version = "easyfhe-disabled"


def _disabled(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch.onnx is disabled in EasyFHE")


def export(*args: Any, **kwargs: Any) -> None:
    _disabled()


def is_in_onnx_export() -> bool:
    return False


register_custom_op_symbolic = _disabled
unregister_custom_op_symbolic = _disabled
select_model_mode_for_export = _disabled
_optimize_trace = _disabled

__all__ = [
    "ExportableModule",
    "InputObserver",
    "ONNXProgram",
    "OnnxExporterError",
    "OperatorExportTypes",
    "TensorProtoDataType",
    "TrainingMode",
    "export",
    "is_in_onnx_export",
]
