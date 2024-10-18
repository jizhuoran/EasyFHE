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
    curr_limbs: int,
    level: int,
    hat_inverse_vec: Optional[Tensor],
    hat_inverse_vec_shoup: Optional[Tensor],
    prod_q_i_mod_q_j: Optional[Tensor],
    primes: Tensor,
    barret_ratio: Tensor,
    barret_k: Tensor,
    beta: int,
    degree: int,
    alpha: int,
    out: Tensor,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.modup_core_(
            out,
            x,
            curr_limbs=curr_limbs,
            level=level,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
        )
    else:
        res = torch.modup_core(
            out,
            x,
            curr_limbs=curr_limbs,
            level=level,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
        )

    return res


def modup(
    x: Tensor,
    curr_limbs: int,
    level: int,
    hat_inverse_vec: Optional[Tensor],
    hat_inverse_vec_shoup: Optional[Tensor],
    prod_q_i_mod_q_j: Optional[Tensor],
    primes: Tensor,
    barret_ratio: Tensor,
    barret_k: Tensor,
    beta: int,
    degree: int,
    alpha: int,
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
            curr_limbs=curr_limbs,
            level=level,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_power_of_roots=param_power_of_roots,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )
    else:
        res = torch.modup(
            out,
            x,
            curr_limbs=curr_limbs,
            level=level,
            hat_inverse_vec=hat_inverse_vec,
            hat_inverse_vec_shoup=hat_inverse_vec_shoup,
            prod_q_i_mod_q_j=prod_q_i_mod_q_j,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            beta=beta,
            degree=degree,
            alpha=alpha,
            param_power_of_roots_shoup=param_power_of_roots_shoup,
            param_power_of_roots=param_power_of_roots,
            inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
        )

    return res


def moddown_core(
    x: Tensor,
    curr_limbs: int,
    level: int,
    alpha: int,
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
            curr_limbs=curr_limbs,
            level=level,
            alpha=alpha,
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
            curr_limbs=curr_limbs,
            level=level,
            alpha=alpha,
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
    curr_limbs: int,
    level: int,
    alpha: int,
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
            curr_limbs=curr_limbs,
            level=level,
            alpha=alpha,
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
            curr_limbs=curr_limbs,
            level=level,
            alpha=alpha,
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


def iNTT(
    x: Tensor,
    curr_limbs: int,
    level: int,
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
            curr_limbs=curr_limbs,
            level=level,
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
            curr_limbs=curr_limbs,
            level=level,
        )
    return res


def innerproduct(
    x: Tensor,
    ax: Tensor,
    bx: Tensor,
    curr_limbs: int,
    alpha: int,
    level: int,
    param_degree: int,
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
            curr_limbs=curr_limbs,
            alpha=alpha,
            level=level,
            param_degree=param_degree,
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
            curr_limbs=curr_limbs,
            alpha=alpha,
            level=level,
            param_degree=param_degree,
            primes=primes,
            barret_ratio=barret_ratio,
            barret_k=barret_k,
            workspace=workspace,
        )
    return res


def keyswitch(
    context_cuda: Context,
    input: Tensor,
    swk_ax: Tensor,
    swk_bx: Tensor,
    curr_limbs: int,
    inplace: bool = False,
) -> Tensor:
    beta = (int)((curr_limbs + 1) / context_cuda.alpha)
    modup_res = modup(
        input,
        curr_limbs=curr_limbs,
        level=context_cuda.level,
        hat_inverse_vec=context_cuda.hat_inverse_vec_modup,
        hat_inverse_vec_shoup=context_cuda.hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j=context_cuda.prod_q_i_mod_q_j_modup[curr_limbs - 1],
        primes=context_cuda.primes,
        barret_ratio=context_cuda.barret_ratio,
        barret_k=context_cuda.barret_k,
        beta=beta,
        degree=context_cuda.degree,
        alpha=context_cuda.alpha,
        param_power_of_roots_shoup=context_cuda.power_of_roots_shoup,
        param_power_of_roots=context_cuda.power_of_roots,
        inverse_power_of_roots_div_two=context_cuda.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context_cuda.inverse_scaled_power_of_roots_div_two,
        out=context_cuda.modup_out,
        inplace=inplace,
    )
    inner_product = innerproduct(
        modup_res,
        ax=swk_ax,
        bx=swk_bx,
        curr_limbs=curr_limbs,
        alpha=context_cuda.alpha,
        level=context_cuda.level,
        param_degree=context_cuda.degree,
        primes=context_cuda.primes,
        barret_ratio=context_cuda.barret_ratio,
        barret_k=context_cuda.barret_k,
        workspace=context_cuda.inner_workspace,
        res=context_cuda.inner_out,
        inplace=inplace,
    )

    sumMult_ax = inner_product[0]
    sumMult_bx = inner_product[1]

    moddown_ax = moddown(
        sumMult_ax,
        curr_limbs=curr_limbs,
        level=context_cuda.level,
        alpha=context_cuda.alpha,
        param_degree=context_cuda.degree,
        param_log_degree=context_cuda.log_degree,
        hat_inverse_vec_moddown=context_cuda.hat_inverse_vec_moddown,
        hat_inverse_vec_shoup_moddown=context_cuda.hat_inverse_vec_shoup_moddown,
        prod_q_i_mod_q_j_moddown=context_cuda.prod_q_i_mod_q_j_moddown,
        prod_inv_moddown=context_cuda.prod_inv_moddown,
        prod_inv_shoup_moddown=context_cuda.prod_inv_shoup_moddown,
        param_primes=context_cuda.primes,
        param_barret_ratio=context_cuda.barret_ratio,
        param_barret_k=context_cuda.barret_k,
        param_power_of_roots_shoup=context_cuda.power_of_roots_shoup,
        param_power_of_roots=context_cuda.power_of_roots,
        inverse_power_of_roots_div_two=context_cuda.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context_cuda.inverse_scaled_power_of_roots_div_two,
        out=context_cuda.moddown_out_ax,
        inplace=inplace,
    )

    moddown_bx = moddown(
        sumMult_bx,
        curr_limbs=curr_limbs,
        level=context_cuda.level,
        alpha=context_cuda.alpha,
        param_degree=context_cuda.degree,
        param_log_degree=context_cuda.log_degree,
        hat_inverse_vec_moddown=context_cuda.hat_inverse_vec_moddown,
        hat_inverse_vec_shoup_moddown=context_cuda.hat_inverse_vec_shoup_moddown,
        prod_q_i_mod_q_j_moddown=context_cuda.prod_q_i_mod_q_j_moddown,
        prod_inv_moddown=context_cuda.prod_inv_moddown,
        prod_inv_shoup_moddown=context_cuda.prod_inv_shoup_moddown,
        param_primes=context_cuda.primes,
        param_barret_ratio=context_cuda.barret_ratio,
        param_barret_k=context_cuda.barret_k,
        param_power_of_roots_shoup=context_cuda.power_of_roots_shoup,
        param_power_of_roots=context_cuda.power_of_roots,
        inverse_power_of_roots_div_two=context_cuda.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context_cuda.inverse_scaled_power_of_roots_div_two,
        out=context_cuda.moddown_out_bx,
        inplace=inplace,
    )

    out = torch.stack((moddown_ax, moddown_bx), dim=0)
    return out
