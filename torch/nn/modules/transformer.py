# mypy: allow-untyped-defs


__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
]


class _DisabledTransformer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torch.nn.modules.transformer is disabled in EasyFHE")


Transformer = _DisabledTransformer
TransformerEncoder = _DisabledTransformer
TransformerDecoder = _DisabledTransformer
TransformerEncoderLayer = _DisabledTransformer
TransformerDecoderLayer = _DisabledTransformer
