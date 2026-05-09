__all__ = []


def __getattr__(name):
    raise RuntimeError("quantization test utilities are disabled in EasyFHE")
