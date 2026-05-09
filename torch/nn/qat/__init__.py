__all__ = []


def __getattr__(name):
    raise RuntimeError("torch.nn.qat is disabled in EasyFHE")
