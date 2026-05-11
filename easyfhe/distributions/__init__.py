class Distribution:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torch.distributions is not available in EasyFHE")


class _DisabledDistribution(Distribution):
    pass


def kl_divergence(*args, **kwargs):
    raise RuntimeError("torch.distributions is not available in EasyFHE")


def register_kl(*args, **kwargs):
    raise RuntimeError("torch.distributions is not available in EasyFHE")


class _Constraints:
    def __getattr__(self, name):
        raise RuntimeError("torch.distributions is not available in EasyFHE")


constraints = _Constraints()


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return type(name, (_DisabledDistribution,), {})


__all__ = ["Distribution", "constraints", "kl_divergence", "register_kl"]

