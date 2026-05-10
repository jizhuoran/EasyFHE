def __getattr__(name):
    raise AttributeError(f"torch.nn.functional.{name} is disabled in EasyFHE")


__all__ = []
