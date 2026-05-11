# mypy: allow-untyped-defs
import easyfhe as torch


def is_available():
    r"""Return whether PyTorch is built with KleidiAI support."""
    return torch._C._has_kleidiai
