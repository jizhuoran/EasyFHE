from collections import OrderedDict
from torch.utils.hooks import RemovableHandle


required = object()
_global_optimizer_pre_hooks: "OrderedDict[int, object]" = OrderedDict()
_global_optimizer_post_hooks: "OrderedDict[int, object]" = OrderedDict()


def register_optimizer_step_pre_hook(hook):
    handle = RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def register_optimizer_step_post_hook(hook):
    handle = RemovableHandle(_global_optimizer_post_hooks)
    _global_optimizer_post_hooks[handle.id] = hook
    return handle


class Optimizer:
    def __init__(self, params=None, defaults=None):
        self.param_groups = []
        self.state = {}
        self.defaults = defaults or {}

    def step(self, closure=None):
        raise RuntimeError("torch.optim optimizers are not available in EasyFHE")

    def zero_grad(self, set_to_none=True):
        return None

    def state_dict(self):
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state_dict):
        self.state = state_dict.get("state", {})
        self.param_groups = state_dict.get("param_groups", [])

