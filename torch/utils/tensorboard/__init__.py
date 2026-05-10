class SummaryWriter:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torch.utils.tensorboard is not available in EasyFHE")


__all__ = ["SummaryWriter"]

