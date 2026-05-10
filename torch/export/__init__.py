"""Export stubs for EasyFHE."""


def _disabled(*args, **kwargs):
    raise RuntimeError("torch.export is disabled in EasyFHE")


export = _disabled
export_for_training = _disabled
load = _disabled
save = _disabled


def register_dataclass(cls=None, **kwargs):
    if cls is None:
        return lambda inner: inner
    return cls


class Dim:
    def __init__(self, name, *, min=None, max=None):
        self.name = name
        self.min = min
        self.max = max


class ExportedProgram:
    pass
