class Timer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torch.utils.benchmark is not available in EasyFHE")


__all__ = ["Timer"]

