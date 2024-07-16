import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
Tensor = torch.Tensor


def vec_add_mod(x: Tensor, y: Tensor, mod: int, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.add_mod_(x, y, mod=mod)
    else:
        res = torch.add_mod(x, y, mod=mod)
    return res


def vec_sub_mod(x: Tensor, y: Tensor, mod: int, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.sub_mod_(x, y, mod=mod)
    else:
        res = torch.sub_mod(x, y, mod=mod)
    return res


def vec_mul_mod(
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
