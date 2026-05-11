from .optimizer import (
    Optimizer,
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
    required,
)


class _DisabledOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torch.optim optimizers are not available in EasyFHE")


Adadelta = Adagrad = Adam = AdamW = Adamax = ASGD = LBFGS = NAdam = RAdam = RMSprop = Rprop = SGD = SparseAdam = _DisabledOptimizer


__all__ = [
    "ASGD",
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "LBFGS",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
    "register_optimizer_step_post_hook",
    "register_optimizer_step_pre_hook",
    "required",
]

