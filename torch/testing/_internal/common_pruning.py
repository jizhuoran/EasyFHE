__all__ = []


def __getattr__(name):
    raise RuntimeError("pruning test utilities are disabled in EasyFHE")
