def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    raise RuntimeError("torch.optim.swa_utils is not available in EasyFHE")

