# mypy: allow-untyped-defs
import sys

from torch.backends import ContextProp, PropModule


def get_cudnn_mode(mode):
    raise RuntimeError("cuDNN RNN is disabled in EasyFHE")


class Unserializable:
    def __init__(self, inner):
        self.inner = inner

    def get(self):
        return self.inner

    def __getstate__(self):
        return "<unserializable>"

    def __setstate__(self, state):
        self.inner = None


def init_dropout_state(dropout, train, dropout_seed, dropout_state):
    if dropout:
        raise RuntimeError("cuDNN RNN is disabled in EasyFHE")
    return None


class CudnnRNNModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)
        self.m.Unserializable = Unserializable
        self.m.get_cudnn_mode = get_cudnn_mode
        self.m.init_dropout_state = init_dropout_state

    fp32_precision = ContextProp(lambda: "none", lambda val: None)


sys.modules[__name__] = CudnnRNNModule(sys.modules[__name__], __name__)
