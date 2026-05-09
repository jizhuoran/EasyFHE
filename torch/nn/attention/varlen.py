# mypy: allow-untyped-defs


def __getattr__(name):
    raise AttributeError(f"torch.nn.attention.varlen.{name} is disabled in EasyFHE")
