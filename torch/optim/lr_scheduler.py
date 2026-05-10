class LRScheduler:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torch.optim.lr_scheduler is not available in EasyFHE")


_LRScheduler = LRScheduler


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return LRScheduler

