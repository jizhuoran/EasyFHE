class Proxy:
    def __init__(self, node=None, *args, **kwargs):
        self.node = node


class Tracer:
    def trace(self, *args, **kwargs):
        raise RuntimeError("torch.fx is disabled in EasyFHE")
