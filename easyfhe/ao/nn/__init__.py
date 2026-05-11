__all__ = []


def __getattr__(name):
    raise RuntimeError("torch.ao.nn is disabled in EasyFHE")
