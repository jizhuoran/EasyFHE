from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _maybe_find_pre_dispatch_tf_mode_for_export() -> None:
    return None


def _fakify_params_buffers(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("torch._export is disabled in EasyFHE")


def register_module_as_pytree_input_node(*args: Any, **kwargs: Any) -> None:
    return None


def deregister_module_as_pytree_input_node(*args: Any, **kwargs: Any) -> None:
    return None


@contextmanager
def _enable_graph_inputs_of_type_nn_module(*args: Any, **kwargs: Any) -> Iterator[None]:
    yield
