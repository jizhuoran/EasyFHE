# mypy: allow-untyped-defs


def __getattr__(name):
    raise AttributeError(f"torch.nn.attention._fa4.{name} is disabled in EasyFHE")
