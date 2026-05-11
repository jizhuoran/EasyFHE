from collections import OrderedDict

from easyfhe.nn.parameter import Parameter


def _addindent(s_, num_spaces):
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    return first + "\n" + s


class Module:
    training = True

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("torch.nn.Module arguments are disabled in EasyFHE")
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")

    def train(self, mode=True):
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_parameters(self, prefix="", recurse=True):
        for name, parameter in self._parameters.items():
            yield f"{prefix}.{name}" if prefix else name, parameter
        if recurse:
            for module_name, module in self._modules.items():
                sub_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_parameters(sub_prefix, recurse=True)

    def buffers(self, recurse=True):
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_buffers(self, prefix="", recurse=True):
        for name, buffer in self._buffers.items():
            yield f"{prefix}.{name}" if prefix else name, buffer
        if recurse:
            for module_name, module in self._modules.items():
                sub_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_buffers(sub_prefix, recurse=True)

    def register_parameter(self, name, param):
        if param is not None and not isinstance(param, Parameter):
            raise TypeError("parameter should be a Parameter or None")
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_buffer(self, name, tensor, persistent=True):
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def add_module(self, name, module):
        if module is not None and not isinstance(module, Module):
            raise TypeError("module should be a Module or None")
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def state_dict(self, *args, **kwargs):
        result = OrderedDict()
        for name, parameter in self.named_parameters():
            result[name] = parameter
        for name, buffer in self.named_buffers():
            result[name] = buffer
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        missing = []
        unexpected = []
        own_state = self.state_dict()
        for name, tensor in state_dict.items():
            if name in own_state:
                own_state[name].copy_(tensor)
            else:
                unexpected.append(name)
        if strict:
            missing = [name for name in own_state if name not in state_dict]
            if missing or unexpected:
                raise RuntimeError(
                    f"Error(s) in loading state_dict: missing={missing}, unexpected={unexpected}"
                )
        return missing, unexpected
