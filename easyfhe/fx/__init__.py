"""Minimal FX compatibility surface for EasyFHE.

FX tracing is intentionally unsupported; this package only keeps imports from
the remaining tensor infrastructure working.
"""

from .graph import Graph
from .graph_module import GraphModule
from .interpreter import Interpreter
from .node import Node
from .proxy import Proxy, Tracer


def symbolic_trace(*args, **kwargs):
    raise RuntimeError("torch.fx is disabled in EasyFHE")


def wrap(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return None


__all__ = [
    "Graph",
    "GraphModule",
    "Interpreter",
    "Node",
    "Proxy",
    "Tracer",
    "symbolic_trace",
    "wrap",
]
