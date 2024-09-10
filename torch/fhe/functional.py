import numpy as np

import torch
from .Ciphertext import Ciphertext
from .context import Context

Tensor = torch.Tensor
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union


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


def modup_core(
    x: Tensor,
    hat_inverse_vec: Optional[Tensor],
    hat_inverse_vec_shoup: Optional[Tensor],
    prod_q_i_mod_q_j: Optional[Tensor],
    primes: Tensor,
    barret_ratio: Tensor,
    barret_k: Tensor,
    beta: int,
    degree: int,
    alpha: int,
    num_moduli_after_modup: int,
    out: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.modup_core_(
            out,
            x,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
            num_moduli_after_modup=num_moduli_after_modup,
        )
    else:
        res = torch.modup_core(
            out,
            x,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
            num_moduli_after_modup=num_moduli_after_modup,
        )

    return res


def modup(
    x: Tensor,
    hat_inverse_vec: Optional[Tensor],
    hat_inverse_vec_shoup: Optional[Tensor],
    prod_q_i_mod_q_j: Optional[Tensor],
    primes: Tensor,
    barret_ratio: Tensor,
    barret_k: Tensor,
    beta: int,
    degree: int,
    alpha: int,
    num_moduli_after_modup: int,
    param_power_of_roots_shoup: Tensor,
    param_power_of_roots: Tensor,
    inverse_power_of_roots_div_two: Tensor,
    inverse_scaled_power_of_roots_div_two: Tensor,
    out: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.modup_(
            out,
            x,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
            num_moduli_after_modup=num_moduli_after_modup,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_power_of_roots=param_power_of_roots,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )
    else:
        res = torch.modup(
            out,
            x,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
            num_moduli_after_modup=num_moduli_after_modup,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_power_of_roots=param_power_of_roots,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )

    return res


def moddown_core(
    x: Tensor,
    target_chain_idx: int,
    param_chain_length: int,
    param_max_num_moduli: int,
    param_degree: int,
    param_log_degree: int,
    hat_inverse_vec_moddown: Optional[Tensor],
    hat_inverse_vec_shoup_moddown: Optional[Tensor],
    prod_q_i_mod_q_j_moddown: Optional[Tensor],
    prod_inv_moddown: Optional[Tensor],
    prod_inv_shoup_moddown: Optional[Tensor],
    param_primes: Tensor,
    param_barret_ratio: Tensor,
    param_barret_k: Tensor,
    out: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.moddown_core_(
            out,
            x,
            target_chain_idx=target_chain_idx,
            param_chain_length=param_chain_length,
            param_max_num_moduli=param_max_num_moduli,
            param_degree=param_degree,
            param_log_degree=param_log_degree,
            hat_inverse_vec_moddown=hat_inverse_vec_moddown,
            hat_inverse_vec_shoup_moddown=hat_inverse_vec_shoup_moddown,
            prod_q_i_mod_q_j_moddown=prod_q_i_mod_q_j_moddown,
            prod_inv_moddown=prod_inv_moddown,
            prod_inv_shoup_moddown=prod_inv_shoup_moddown,
            param_primes=param_primes,
            param_barret_ratio=param_barret_ratio,
            param_barret_k=param_barret_k,
        )
    else:
        res = torch.moddown_core(
            out,
            x,
            target_chain_idx=target_chain_idx,
            param_chain_length=param_chain_length,
            param_max_num_moduli=param_max_num_moduli,
            param_degree=param_degree,
            param_log_degree=param_log_degree,
            hat_inverse_vec_moddown=hat_inverse_vec_moddown,
            hat_inverse_vec_shoup_moddown=hat_inverse_vec_shoup_moddown,
            prod_q_i_mod_q_j_moddown=prod_q_i_mod_q_j_moddown,
            prod_inv_moddown=prod_inv_moddown,
            prod_inv_shoup_moddown=prod_inv_shoup_moddown,
            param_primes=param_primes,
            param_barret_ratio=param_barret_ratio,
            param_barret_k=param_barret_k,
        )

    return res


def moddown(
    x: Tensor,
    target_chain_idx: int,
    param_chain_length: int,
    param_max_num_moduli: int,
    param_degree: int,
    param_log_degree: int,
    hat_inverse_vec_moddown: Optional[Tensor],
    hat_inverse_vec_shoup_moddown: Optional[Tensor],
    prod_q_i_mod_q_j_moddown: Optional[Tensor],
    prod_inv_moddown: Optional[Tensor],
    prod_inv_shoup_moddown: Optional[Tensor],
    param_primes: Tensor,
    param_barret_ratio: Tensor,
    param_barret_k: Tensor,
    param_power_of_roots_shoup: Tensor,
    param_power_of_roots: Tensor,
    inverse_power_of_roots_div_two: Tensor,
    inverse_scaled_power_of_roots_div_two: Tensor,
    out: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.moddown_(
            out,
            x,
            target_chain_idx=target_chain_idx,
            param_chain_length=param_chain_length,
            param_max_num_moduli=param_max_num_moduli,
            param_degree=param_degree,
            param_log_degree=param_log_degree,
            hat_inverse_vec_moddown=hat_inverse_vec_moddown,
            hat_inverse_vec_shoup_moddown=hat_inverse_vec_shoup_moddown,
            prod_q_i_mod_q_j_moddown=prod_q_i_mod_q_j_moddown,
            prod_inv_moddown=prod_inv_moddown,
            prod_inv_shoup_moddown=prod_inv_shoup_moddown,
            param_primes=param_primes,
            param_barret_ratio=param_barret_ratio,
            param_barret_k=param_barret_k,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_power_of_roots=param_power_of_roots,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )
    else:
        res = torch.moddown(
            out,
            x,
            target_chain_idx=target_chain_idx,
            param_chain_length=param_chain_length,
            param_max_num_moduli=param_max_num_moduli,
            param_degree=param_degree,
            param_log_degree=param_log_degree,
            hat_inverse_vec_moddown=hat_inverse_vec_moddown,
            hat_inverse_vec_shoup_moddown=hat_inverse_vec_shoup_moddown,
            prod_q_i_mod_q_j_moddown=prod_q_i_mod_q_j_moddown,
            prod_inv_moddown=prod_inv_moddown,
            prod_inv_shoup_moddown=prod_inv_shoup_moddown,
            param_primes=param_primes,
            param_barret_ratio=param_barret_ratio,
            param_barret_k=param_barret_k,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_power_of_roots=param_power_of_roots,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )

    return res


def NTT(
    x: Tensor,
    start_prime_idx: int,
    batch: int,
    param_degree: int,
    param_power_of_roots_shoup: Tensor,
    param_primes: Tensor,
    param_power_of_roots: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.NTT_(
            x,
            start_prime_idx=start_prime_idx,
            batch=batch,
            param_degree=param_degree,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_primes=param_primes,
            param_power_of_roots=param_power_of_roots,
        )
    else:
        res = torch.NTT(
            x,
            start_prime_idx=start_prime_idx,
            batch=batch,
            param_degree=param_degree,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_primes=param_primes,
            param_power_of_roots=param_power_of_roots,
        )
    return res


def NTT_except_some_range(
    x: Tensor,
    start_prime_idx: int,
    batch: int,
    param_degree: int,
    excluded_range_start: int,
    excluded_range_size: int,
    param_power_of_roots_shoup: Tensor,
    param_primes: Tensor,
    param_power_of_roots: Tensor,
) -> Tensor:
    res = torch.NTT_except_some_range_cuda(
        x,
        start_prime_idx=start_prime_idx,
        batch=batch,
        param_degree=param_degree,
        excluded_range_start=excluded_range_start,
        excluded_range_size=excluded_range_size,
        param_power_of_roots_shoup=param_power_of_roots_shoup,
        param_primes=param_primes,
        param_power_of_roots=param_power_of_roots,
    )
    return res


def iNTT(
    x: Tensor,
    start_prime_idx: int,
    batch: int,
    param_degree: int,
    inverse_power_of_roots_div_two: Tensor,
    param_primes: Tensor,
    inverse_scaled_power_of_roots_div_two: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.iNTT_(
            x,
            start_prime_idx=start_prime_idx,
            batch=batch,
            param_degree=param_degree,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            param_primes=param_primes,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )
    else:
        res = torch.iNTT(
            x,
            start_prime_idx=start_prime_idx,
            batch=batch,
            param_degree=param_degree,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            param_primes=param_primes,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )
    return res


def innerproduct(
    x: Tensor,
    ax: Tensor,
    bx: Tensor,
    param_degree: int,
    param_max_num_moduli: int,
    primes: Tensor,
    barret_ratio: Tensor,
    barret_k: Tensor,
    workspace: Tensor,
    res: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.innerproduct_(
            res,
            x,
            ax=ax,
            bx=bx,
            param_degree=param_degree,
            param_max_num_moduli=param_max_num_moduli,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            workspace=workspace,
        )
    else:
        res = torch.innerproduct(
            res,
            x,
            ax=ax,
            bx=bx,
            param_degree=param_degree,
            param_max_num_moduli=param_max_num_moduli,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            workspace=workspace,
        )
    return res
