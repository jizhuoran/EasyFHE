# mypy: allow-untyped-defs
import easyfhe as torch
import easyfhe._C._lazy_ts_backend


def init():
    """Initializes the lazy Torchscript backend"""
    torch._C._lazy_ts_backend._init()
