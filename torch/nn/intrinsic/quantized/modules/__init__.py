# mypy: allow-untyped-defs


__all__: list[str] = []


def __getattr__(name):
    raise AttributeError(
        f"torch.nn.intrinsic.quantized.modules.{name} is disabled in EasyFHE"
    )
