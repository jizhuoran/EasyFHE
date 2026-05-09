__all__ = []


def __getattr__(name):
    raise RuntimeError("torch.nn.quantizable is disabled in EasyFHE")
