def __getattr__(name):
    raise AttributeError(f"torch.nn.init.{name} is disabled in EasyFHE")


__all__ = []
