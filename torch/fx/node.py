from typing import Any

Target = Any
Argument = Any
_side_effectful_functions = set()


class Node:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.meta = {}
        self.target = kwargs.get("target", None)
        self.graph = kwargs.get("graph", None)


def map_aggregate(a, fn):
    if isinstance(a, tuple):
        return tuple(map_aggregate(x, fn) for x in a)
    if isinstance(a, list):
        return [map_aggregate(x, fn) for x in a]
    if isinstance(a, dict):
        return {k: map_aggregate(v, fn) for k, v in a.items()}
    return fn(a)


def map_arg(a, fn):
    return map_aggregate(a, fn)


def has_side_effect(fn):
    _side_effectful_functions.add(fn)
    return fn
