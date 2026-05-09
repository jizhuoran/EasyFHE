# mypy: allow-untyped-defs


def __getattr__(name):
    raise AttributeError(f"torch.nn.attention._utils.{name} is disabled in EasyFHE")
