import torch

Tensor = torch.Tensor


def add_mod(x: Tensor, y: Tensor, mod, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.add_mod_(x, y, mod=mod)
    else:
        res = torch.add_mod(x, y, mod=mod)
    return res


def sub_mod(x: Tensor, y: Tensor, mod, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.sub_mod_(x, y, mod=mod)
    else:
        res = torch.sub_mod(x, y, mod=mod)
    return res


def mul_mod(
    x: Tensor,
    y: Tensor,
    mod,
    barret_mu,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.mul_mod_(x, y, mod=mod, barret_mu=barret_mu)
    else:
        res = torch.mul_mod(x, y, mod=mod, barret_mu=barret_mu)
    return res
