__all__ = ["nn", "ns", "pruning", "quantization"]


def _disabled(name):
    raise RuntimeError(f"torch.ao.{name} is disabled in EasyFHE")


def __getattr__(name):
    if name in __all__:
        _disabled(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
