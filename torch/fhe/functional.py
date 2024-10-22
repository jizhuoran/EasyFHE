import torch
from typing import Optional
from .context import Context
import numpy as np

Tensor = torch.Tensor


def vec_add_mod(x, y, MOD):
    return [int((int(a) + int(b)) % MOD) for a, b in zip(x, y)]


def vec_sub_mod(x, y, MOD):
    return [int((int(a) - int(b)) % MOD) for a, b in zip(x, y)]


def vec_mul_mod(x, y, MOD):
    return [int((int(a) * int(b)) % MOD) for a, b in zip(x, y)]


def cv_convert(func):
    def wrapper(*args, **kw):
        args_list = list(args)
        for i in range(len(args_list)):
            if isinstance(args_list[i], np.ndarray):
                args_list[i] = torch.from_numpy(args_list[i])
                args_list[i] = args_list[i].cuda()
        new_args = tuple(args_list)
        res = func(*new_args, **kw)
        return res.cpu().numpy()

    return wrapper


def cv_check(x, modulus, cur_limbs):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(modulus, torch.Tensor):
        modulus = modulus.cpu().numpy()
    assert len(x.shape) == 2
    for l in range(x.shape[0]):
        for i in range(x.shape[1]):
            if x[l][i] < 0 or x[l][i] >= modulus[l]:
                print(l, i, x[l][i], modulus[l])
                # assert False


def gen_scalar_tensor(scalar, modulus, cur_limbs):
    return torch.from_numpy(
        np.array(
            [int(int(scalar) % int(modulus[l])) for l in range(cur_limbs)],
            dtype=np.uint64,
        )
    ).cuda()


@cv_convert
def cv_neg(x, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.neg_mod_(x, x, modulus, cur_limbs=cur_limbs)
    else:
        return torch.neg_mod(x, x, modulus, cur_limbs=cur_limbs)


@cv_convert
def cv_add(x, y, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.add_mod_(x, y, modulus, cur_limbs=cur_limbs)
    else:
        return torch.add_mod(x, y, modulus, cur_limbs=cur_limbs)


@cv_convert
def cv_sub(x, y, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.sub_mod_(x, y, modulus, cur_limbs=cur_limbs)
    else:
        return torch.sub_mod(x, y, modulus, cur_limbs=cur_limbs)


@cv_convert
def cv_mul(x, y, modulus, barret_mu, cur_limbs, inplace=False):
    if inplace:
        return torch.mul_mod_(x, y, modulus, barret_mu, cur_limbs=cur_limbs)
    else:
        return torch.mul_mod(x, y, modulus, barret_mu, cur_limbs=cur_limbs)


@cv_convert
def cv_add_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.add_scalar_mod_(x, scalar, modulus, cur_limbs=cur_limbs)
    else:
        return torch.add_scalar_mod(x, scalar, modulus, cur_limbs=cur_limbs)


@cv_convert
def cv_sub_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.sub_scalar_mod_(x, scalar, modulus, cur_limbs=cur_limbs)
    else:
        return torch.sub_scalar_mod(x, scalar, modulus, cur_limbs=cur_limbs)


@cv_convert
def cv_mul_scalar(x, scalar, modulus, barret_mu, cur_limbs, inplace=False):
    if inplace:
        return torch.mul_scalar_mod_(x, scalar, modulus, barret_mu, cur_limbs=cur_limbs)
    else:
        return torch.mul_scalar_mod(x, scalar, modulus, barret_mu, cur_limbs=cur_limbs)

def cv_modup(
    x: Tensor,
    curr_limbs: int,
    context: Context,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.modup_(
            context.modup_out,
            x,
            curr_limbs=curr_limbs,
            level=context.level,
            hat_inverse_vec=context.hat_inverse_vec_modup,
            hat_inverse_vec_shoup=context.hat_inverse_vec_shoup_modup,
            prod_q_i_mod_q_j=context.prod_q_i_mod_q_j_modup[curr_limbs-1],
            primes=context.primes,
            barret_ratio=context.barret_ratio,
            barret_k=context.barret_k,
            beta=context.beta,
            degree=context.degree,
            alpha=context.alpha,
            param_power_of_roots_shoup=context.power_of_roots_shoup,
            param_power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        )
    else:
        res = torch.modup(
            context.modup_out,
            x,
            curr_limbs=curr_limbs,
            level=context.level,
            hat_inverse_vec=context.hat_inverse_vec_modup,
            hat_inverse_vec_shoup=context.hat_inverse_vec_shoup_modup,
            prod_q_i_mod_q_j=context.prod_q_i_mod_q_j_modup[curr_limbs-1],
            primes=context.primes,
            barret_ratio=context.barret_ratio,
            barret_k=context.barret_k,
            beta=context.beta,
            degree=context.degree,
            alpha=context.alpha,
            param_power_of_roots_shoup=context.power_of_roots_shoup,
            param_power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        )

    return res


def cv_moddown(
    x: Tensor,
    curr_limbs: int,
    context: Context,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.moddown_(
            context.moddown_out_ax,
            x,
            curr_limbs=curr_limbs,
            level=context.level,
            alpha=context.alpha,
            param_degree=context.degree,
            param_log_degree=context.log_degree,
            hat_inverse_vec_moddown=context.hat_inverse_vec_moddown,
            hat_inverse_vec_shoup_moddown=context.hat_inverse_vec_shoup_moddown,
            prod_q_i_mod_q_j_moddown=context.prod_q_i_mod_q_j_moddown,
            prod_inv_moddown=context.prod_inv_moddown,
            prod_inv_shoup_moddown=context.prod_inv_shoup_moddown,
            param_primes=context.primes,
            param_barret_ratio=context.barret_ratio,
            param_barret_k=context.barret_k,
            param_power_of_roots_shoup=context.power_of_roots_shoup,
            param_power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        )
    else:
        res = torch.moddown(
            context.moddown_out_ax,
            x,
            curr_limbs=curr_limbs,
            level=context.level,
            alpha=context.alpha,
            param_degree=context.degree,
            param_log_degree=context.log_degree,
            hat_inverse_vec_moddown=context.hat_inverse_vec_moddown,
            hat_inverse_vec_shoup_moddown=context.hat_inverse_vec_shoup_moddown,
            prod_q_i_mod_q_j_moddown=context.prod_q_i_mod_q_j_moddown,
            prod_inv_moddown=context.prod_inv_moddown,
            prod_inv_shoup_moddown=context.prod_inv_shoup_moddown,
            param_primes=context.primes,
            param_barret_ratio=context.barret_ratio,
            param_barret_k=context.barret_k,
            param_power_of_roots_shoup=context.power_of_roots_shoup,
            param_power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
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


def cv_innerproduct(
    x: Tensor,
    curr_limbs: int,
    context_cuda: Context,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.innerproduct_(
            context_cuda.inner_out,
            x,
            ax=context_cuda.swk_ax_cuda,
            bx=context_cuda.swk_bx_cuda,
            curr_limbs=curr_limbs,
            alpha=context_cuda.alpha,
            level=context_cuda.level,
            param_degree=context_cuda.degree,
            primes=context_cuda.primes,
            barret_ratio=context_cuda.barret_ratio,
            barret_k=context_cuda.barret_k,
            workspace=context_cuda.inner_workspace,
        )
    else:
        res = torch.innerproduct(
            context_cuda.inner_out,
            x,
            ax=context_cuda.swk_ax_cuda,
            bx=context_cuda.swk_bx_cuda,
            curr_limbs=curr_limbs,
            alpha=context_cuda.alpha,
            level=context_cuda.level,
            param_degree=context_cuda.degree,
            primes=context_cuda.primes,
            barret_ratio=context_cuda.barret_ratio,
            barret_k=context_cuda.barret_k,
            workspace=context_cuda.inner_workspace,
        )
    return res


def cv_keyswitch(
    input: Tensor,
    cur_limbs: int,
    context_cuda: Context,
    inplace: bool = False,
) -> Tensor:
    true_beta = int((cur_limbs + (context_cuda.K - 1)) / context_cuda.K)
    context_cuda.beta = true_beta
    modup_res = cv_modup(
        input,
        curr_limbs=cur_limbs,
        context=context_cuda,
        inplace=inplace,
    )
    inner_product = cv_innerproduct(
        modup_res,
        cur_limbs,
        context_cuda,
        inplace=inplace,
    )

    sumMult_ax = inner_product[0]
    sumMult_bx = inner_product[1]

    moddown_ax = cv_moddown(
        sumMult_ax,
        curr_limbs=cur_limbs,
        context=context_cuda,
        inplace=False,
    )

    moddown_bx = cv_moddown(
        sumMult_bx,
        curr_limbs=cur_limbs,
        context=context_cuda,
        inplace=False,
    )

    out = torch.stack((moddown_ax, moddown_bx), dim=0)
    return out