__all__ = []


def __getattr__(name):
    raise RuntimeError("torch.ao.pruning is disabled in EasyFHE")
