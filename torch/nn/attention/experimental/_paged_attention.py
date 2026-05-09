# mypy: allow-untyped-defs


def __getattr__(name):
    raise AttributeError(
        f"torch.nn.attention.experimental._paged_attention.{name} is disabled in EasyFHE"
    )
