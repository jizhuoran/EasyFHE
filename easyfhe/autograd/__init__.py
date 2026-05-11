"""Minimal autograd compatibility layer for EasyFHE.

EasyFHE keeps PyTorch's tensor wrapper and inference-mode machinery, but the
backward engine is intentionally disabled.
"""

from __future__ import annotations

from typing import Any

import easyfhe as torch

from . import function as function
from . import graph as graph
from .function import (
    BackwardCFunction,
    Function,
    FunctionCtx,
    InplaceFunction,
    NestedIOFunction,
    once_differentiable,
)
from .grad_mode import (
    enable_grad,
    enforce_grad_layout_policy,
    inference_mode,
    no_grad,
    set_grad_enabled,
    set_multithreading_enabled,
)
from .variable import Variable


__all__ = [
    "BackwardCFunction",
    "Function",
    "FunctionCtx",
    "InplaceFunction",
    "NestedIOFunction",
    "Variable",
    "backward",
    "detect_anomaly",
    "enable_grad",
    "enforce_grad_layout_policy",
    "grad",
    "gradcheck",
    "gradgradcheck",
    "inference_mode",
    "no_grad",
    "once_differentiable",
    "set_detect_anomaly",
    "set_grad_enabled",
    "set_multithreading_enabled",
]


def _disabled(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "torch.autograd is disabled in EasyFHE. "
        "Prepare gradients outside EasyFHE or run tensor-only FHE inference."
    )


backward = _disabled
grad = _disabled
gradcheck = _disabled
gradgradcheck = _disabled


class detect_anomaly:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disabled()


class set_detect_anomaly:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disabled()


def variable(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "torch.autograd.variable(...) is disabled; use torch.tensor(...) instead"
    )


variable.Variable = Variable  # type: ignore[attr-defined]


if not torch._C._autograd_init():
    raise RuntimeError("autograd compatibility initialization failed")


def _is_checkpoint_valid() -> bool:
    return False


def _register_py_tensor_class_for_device(device: str, cls: type) -> None:
    if not isinstance(cls, type):
        raise RuntimeError("cls isn't a typeinfo object")
    torch._C._register_py_class_for_device(device, cls)


is_multithreading_enabled = torch._C._is_multithreading_enabled
is_view_replay_enabled = torch._C._is_view_replay_enabled


try:
    from easyfhe._C._autograd import (
        _add_metadata_json,
        _disable_profiler,
        _disable_profiler_legacy,
        _enable_profiler,
        _enable_profiler_legacy,
        _enable_record_function,
        _get_sequence_nr,
        _is_kineto_stopped,
        _kineto_step,
        _KinetoEvent,
        _pop_saved_tensors_default_hooks,
        _prepare_profiler,
        _profiler_enabled,
        _ProfilerResult,
        _push_saved_tensors_default_hooks,
        _record_function_with_args_enter,
        _record_function_with_args_exit,
        _set_empty_test_observer,
        _supported_activities,
        _toggle_collection_dynamic,
        DeviceType,
        kineto_available,
        ProfilerEvent,
        SavedTensor,
    )
    from easyfhe._C._profiler import ProfilerActivity, ProfilerConfig, ProfilerState
except (AttributeError, ImportError):
    def _profiler_disabled(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("torch.autograd profiler bindings are disabled in EasyFHE")

    _add_metadata_json = _profiler_disabled
    _disable_profiler = _profiler_disabled
    _disable_profiler_legacy = _profiler_disabled
    _enable_profiler = _profiler_disabled
    _enable_profiler_legacy = _profiler_disabled
    _enable_record_function = _profiler_disabled
    _get_sequence_nr = lambda: 0
    _is_kineto_stopped = lambda: True
    _kineto_step = lambda: 0
    _pop_saved_tensors_default_hooks = _profiler_disabled
    _prepare_profiler = _profiler_disabled
    _profiler_enabled = lambda: False
    _push_saved_tensors_default_hooks = _profiler_disabled
    _record_function_with_args_enter = _profiler_disabled
    _record_function_with_args_exit = _profiler_disabled
    _set_empty_test_observer = lambda *args, **kwargs: None
    _supported_activities = lambda: set()
    _toggle_collection_dynamic = _profiler_disabled
    kineto_available = lambda: False

    class _KinetoEvent:  # type: ignore[no-redef]
        pass

    class _ProfilerResult:  # type: ignore[no-redef]
        pass

    class ProfilerEvent:  # type: ignore[no-redef]
        pass

    class SavedTensor:  # type: ignore[no-redef]
        pass

    DeviceType = torch._C.DeviceType
    from easyfhe._C._profiler import ProfilerActivity, ProfilerConfig, ProfilerState
