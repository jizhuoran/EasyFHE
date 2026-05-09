# mypy: allow-untyped-defs


def _disabled(*args, **kwargs):
    raise RuntimeError("torch.fx is disabled in EasyFHE fast build")


symbolic_trace = _disabled

__all__ = ["symbolic_trace"]
