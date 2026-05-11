__all__ = []


def __getattr__(name):
    raise RuntimeError("torch.ao.ns is disabled in EasyFHE")
