# mypy: allow-untyped-defs


def _disabled(*args, **kwargs):
    raise RuntimeError("torch.nn.attention.bias is disabled in EasyFHE")


class CausalBias:
    def __init__(self, *args, **kwargs):
        _disabled()


class CausalVariant:
    UPPER_LEFT = "upper_left"
    LOWER_RIGHT = "lower_right"


causal_lower_right = _disabled
causal_upper_left = _disabled
_calculate_scale = _disabled
