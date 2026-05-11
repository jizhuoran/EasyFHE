__all__ = []


def __getattr__(name):
    raise RuntimeError("torch.quantization is disabled in EasyFHE")
