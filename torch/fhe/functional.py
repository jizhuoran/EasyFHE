import torch

Tensor = torch.Tensor


def add_mod(x: Tensor, y: Tensor, mod: int, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.add_mod_(x, y, mod=mod)
    else:
        res = torch.add_mod(x, y, mod=mod)
    return res


def sub_mod(x: Tensor, y: Tensor, mod: int, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.sub_mod_(x, y, mod=mod)
    else:
        res = torch.sub_mod(x, y, mod=mod)
    return res


def mul_mod(
    x: Tensor,
    y: Tensor,
    mod: int,
    barret_mu0: int,
    barret_mu1: int,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.mul_mod_(
            x, y, mod=mod, barret_mu0=barret_mu0, barret_mu1=barret_mu1
        )
    else:
        res = torch.mul_mod(x, y, mod=mod, barret_mu0=barret_mu0, barret_mu1=barret_mu1)
    return res
