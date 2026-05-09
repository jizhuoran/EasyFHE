__all__ = []


def __getattr__(name):
    raise RuntimeError("context parallel attention is disabled in EasyFHE")
