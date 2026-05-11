from easyfhe.nn.modules.module import Module as Module


def __getattr__(name):
    if name == "Module":
        return Module
    raise AttributeError(f"torch.nn.modules.{name} is disabled in EasyFHE")


__all__ = ["Module"]
