import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context

Tensor = torch.Tensor


def cv_add(x, y, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.add_mod_(x, y, modulus, L=cur_limbs)
    else:
        return torch.add_mod(x, y, modulus, L=cur_limbs)


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


def add_scalar_mod(x: Tensor, y, mod, inplace: bool = False) -> Tensor:
    if inplace:
        res = torch.add_scalar_mod_(x, y, mod=mod)
    else:
        res = torch.add_scalar_mod(x, y, mod=mod)
    return res


def mul_scalar_mod(
    x: Tensor,
    y,
    mod,
    barret_mu,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.mul_scalar_mod_(x, y, mod=mod, barret_mu=barret_mu)
    else:
        res = torch.mul_scalar_mod(x, y, mod=mod, barret_mu=barret_mu)
    return res


def automorphism(input: Tensor, index: Tensor) -> Tensor:
    return torch.automorphism(input, index)


def mod_switch(input: Tensor, new_modulus, old_modulus) -> Tensor:
    return torch.mod_switch(input, new_modulus, old_modulus)


def NTT(input: Tensor, omega_table: Tensor, mod, barret_mu) -> Tensor:
    return torch.NTT(input, omega_table, mod=mod, barret_mu=barret_mu)


def INTT(input: Tensor, omega_table: Tensor, mod, barret_mu, n_inverse) -> Tensor:
    return torch.NTT(
        input, omega_table, mod=mod, barret_mu=barret_mu, n_inverse=n_inverse
    )
