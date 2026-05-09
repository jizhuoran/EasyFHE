__all__ = []


def __getattr__(name):
    raise RuntimeError("torch.ao.quantization is disabled in EasyFHE")
