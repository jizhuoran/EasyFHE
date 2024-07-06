import torch
Tensor = torch.Tensor

def add_mod(x: Tensor, y: Tensor, mod: int, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.add_mod_(x, y, mod=mod)
    else:
        res = torch.add_mod(x, y, mod=mod)
    return res