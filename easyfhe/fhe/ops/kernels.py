from __future__ import annotations

import easyfhe as torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import Context

Tensor = torch.Tensor

def cv_check(x, modulus, cur_limbs):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(modulus, torch.Tensor):
        modulus = modulus.cpu().numpy()
    if len(x.shape) != 2:
        raise ValueError(f"cv_check: expected a 2D array, got shape={x.shape}")
    for l in range(x.shape[0]):
        for i in range(x.shape[1]):
            if x[l][i] < 0 or x[l][i] >= modulus[l]:
                print(l, i, x[l][i], modulus[l])


def gen_scalar_tensor(scalar, modulus, cur_limbs):
    if isinstance(scalar, int):
        scalar_list = [int(int(scalar) % int(modulus[l])) for l in range(cur_limbs)]
    else:
        scalar_list = [int(int(scalar[l]) % int(modulus[l])) for l in range(cur_limbs)]
    return torch.from_numpy(np.array(scalar_list, dtype=np.uint64))


def cv_neg(x, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.neg_mod_(x, x, modulus, cur_limbs=cur_limbs)
    else:
        return torch.neg_mod(x, x, modulus, cur_limbs=cur_limbs)


def cv_add(x, y, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.add_mod_(x, y, modulus, cur_limbs=cur_limbs)
    else:
        return torch.add_mod(x, y, modulus, cur_limbs=cur_limbs)


def cv_sub(x, y, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.sub_mod_(x, y, modulus, cur_limbs=cur_limbs)
    else:
        return torch.sub_mod(x, y, modulus, cur_limbs=cur_limbs)


def cv_mul(x, y, modulus, barret_mu, cur_limbs, inplace=False):
    if inplace:
        return torch.mul_mod_(x, y, modulus, barret_mu, cur_limbs=cur_limbs)
    else:
        return torch.mul_mod(x, y, modulus, barret_mu, cur_limbs=cur_limbs)


def cv_add_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.add_scalar_mod_(x, scalar, modulus, cur_limbs=cur_limbs)
    else:
        return torch.add_scalar_mod(x, scalar, modulus, cur_limbs=cur_limbs)


def cv_sub_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.sub_scalar_mod_(x, scalar, modulus, cur_limbs=cur_limbs)
    else:
        return torch.sub_scalar_mod(x, scalar, modulus, cur_limbs=cur_limbs)


def cv_mul_scalar(x, scalar, modulus, barret_mu, cur_limbs, inplace=False):
    if inplace:
        return torch.mul_scalar_mod_(x, scalar, modulus, barret_mu, cur_limbs=cur_limbs)
    else:
        return torch.mul_scalar_mod(x, scalar, modulus, barret_mu, cur_limbs=cur_limbs)


def cv_modup(
    x: Tensor,
    curr_limbs: int,
    context: Context,
) -> Tensor:
    beta = (curr_limbs + context.alpha - 1) // context.alpha
    return torch.modup(
        x,
        curr_limbs=curr_limbs,
        L=context.L,
        beta=beta,
        degree=context.N,
        alpha=context.alpha,
        hat_inverse_vec=context.hat_inverse_vec_modup,
        hat_inverse_vec_shoup=context.hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j=context.prod_q_i_mod_q_j_modup[curr_limbs - 1],
        primes=context.primes,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
        inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
    ).reshape(-1, context.N)



def cv_moddown(
    x: Tensor,
    curr_limbs: int,
    context: Context,
) -> Tensor:
    return torch.moddown(
        x,
        curr_limbs=curr_limbs,
        L=context.L,
        sizeP=context.K,
        N=context.N,
        logN=context.logN,
        hat_inverse_vec_moddown=context.hat_inverse_vec_moddown,
        hat_inverse_vec_shoup_moddown=context.hat_inverse_vec_shoup_moddown,
        prod_q_i_mod_q_j_moddown=context.prod_q_i_mod_q_j_moddown,
        prod_inv_moddown=context.prod_inv_moddown,
        prod_inv_shoup_moddown=context.prod_inv_shoup_moddown,
        primes=context.primes,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
        inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
    ).reshape(-1, context.N)


# def NTT(
#     x: Tensor,
#     start_prime_idx: int,
#     batch: int,
#     param_degree: int,
#     param_power_of_roots_shoup: Tensor,
#     param_primes: Tensor,
#     param_power_of_roots: Tensor,
#     inplace: bool = False,
# ) -> Tensor:
#     if inplace:
#         res = torch.NTT_(
#             x,
#             start_prime_idx=start_prime_idx,
#             batch=batch,
#             param_degree=param_degree,
#             param_power_of_roots_shoup=param_power_of_roots_shoup,
#             param_primes=param_primes,
#             param_power_of_roots=param_power_of_roots,
#         )
#     else:
#         res = torch.NTT(
#             x,
#             start_prime_idx=start_prime_idx,
#             batch=batch,
#             param_degree=param_degree,
#             param_power_of_roots_shoup=param_power_of_roots_shoup,
#             param_primes=param_primes,
#             param_power_of_roots=param_power_of_roots,
#         )
#     return res


# def iNTT(
#     x: Tensor,
#     curr_limbs: int,
#     level: int,
#     start_prime_idx: int,
#     batch: int,
#     param_degree: int,
#     inverse_power_of_roots_div_two: Tensor,
#     param_primes: Tensor,
#     inverse_scaled_power_of_roots_div_two: Tensor,
#     inplace: bool = False,
# ) -> Tensor:
#     if inplace:
#         res = torch.iNTT_(
#             x,
#             start_prime_idx=start_prime_idx,
#             batch=batch,
#             param_degree=param_degree,
#             inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
#             param_primes=param_primes,
#             inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
#             curr_limbs=curr_limbs,
#             level=level,
#         )
#     else:
#         res = torch.iNTT(
#             x,
#             start_prime_idx=start_prime_idx,
#             batch=batch,
#             param_degree=param_degree,
#             inverse_power_of_roots_div_two=inverse_power_of_roots_div_two,
#             param_primes=param_primes,
#             inverse_scaled_power_of_roots_div_two=inverse_scaled_power_of_roots_div_two,
#             curr_limbs=curr_limbs,
#             level=level,
#         )
#     return res


def cv_innerproduct(
    x: Tensor,
    curr_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context
) -> Tensor:
    x.reshape(-1)
    res = torch.innerproduct(
        context.inner_out,
        x,
        bx=swk_bx,
        ax=swk_ax,
        curr_limbs=curr_limbs,
        alpha= context.alpha,
        special_mod_start = special_mod_start,
        L=context.L,
        N=context.N,
        primes=context.primes,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        workspace=context.inner_workspace,
    )
    return res.reshape(2, -1, context.N)


def cv_keyswitch(
    input: Tensor,
    cur_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
) -> list:
    modup_res = cv_modup(
        input,
        curr_limbs=cur_limbs,
        context=context
    )
    inner_product = cv_innerproduct(
        modup_res.reshape(-1),
        cur_limbs,
        special_mod_start,
        swk_bx,
        swk_ax,
        context
    )


    moddown_bx = cv_moddown(
        inner_product[0],
        curr_limbs=cur_limbs,
        context=context
    )

    moddown_ax = cv_moddown(
        inner_product[1],
        curr_limbs=cur_limbs,
        context=context
    )

    return [moddown_bx, moddown_ax]


def cv_drop_last_element_and_scale(
    input: Tensor,
    cur_limbs: int,
    l: int,
    context: Context,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        rescale = torch.drop_last_element_and_scale_(
            context.rescale_out,
            input,
            curr_limbs=cur_limbs,
            l=l,
            L=context.L,
            N=context.N,
            primes=context.primes,
            barret_ratio=context.barret_ratio,
            barret_k=context.barret_k,
            power_of_roots_shoup=context.power_of_roots_shoup,
            power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
            qlql_inv_mod_ql_div_ql_mod_q=context.qlql_inv_mod_ql_div_ql_mod_q,
            qlql_inv_mod_ql_div_ql_mod_q_shoup=context.qlql_inv_mod_ql_div_ql_mod_q_shoup,
            q_inv_mod_q=context.q_inv_mod_q,
            q_inv_mod_q_shoup=context.q_inv_mod_q_shoup,
        )
    else:
        rescale = torch.drop_last_element_and_scale(
            context.rescale_out,
            input,
            curr_limbs=cur_limbs,
            l=l,
            L=context.L,
            N=context.N,
            primes=context.primes,
            barret_ratio=context.barret_ratio,
            barret_k=context.barret_k,
            power_of_roots_shoup=context.power_of_roots_shoup,
            power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
            qlql_inv_mod_ql_div_ql_mod_q=context.qlql_inv_mod_ql_div_ql_mod_q,
            qlql_inv_mod_ql_div_ql_mod_q_shoup=context.qlql_inv_mod_ql_div_ql_mod_q_shoup,
            q_inv_mod_q=context.q_inv_mod_q,
            q_inv_mod_q_shoup=context.q_inv_mod_q_shoup,
        )

    return rescale.reshape(-1, context.N)


def cv_automorphism_transform(
    input: Tensor,
    cur_limbs: int,
    i: int,
    context: Context
) -> Tensor:
    if i < 0:
        raise ValueError("i should be non-negative")
    return torch.automorphism_transform(
        input, l=cur_limbs, N=context.N, precomp_vec=context.get_precompute_auto(i)
    )


def cv_mul_by_monomial(
    input: Tensor,
    l: int,
    monomialDeg: int,
    context: Context,
) -> None:
    # "monomial only supports inplace operation"
    torch.mul_by_monomial_(
        input,
        l=l,
        N=context.N,
        M=context.M,
        monomialDeg=monomialDeg,
        L=context.L,
        primes=context.primes,
        inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
    )

def cipher_fused_pairwise_mac(
    ctx_bxs, ctx_axs, ptx_bxs, primes, barret_mu, len_ctxs, cur_limb, N
):
    res = torch.fused_pairwise_mac(
        ctx_bxs,
        ctx_axs,
        ptx_bxs,
        primes,
        barret_mu,
        len_ctxs,
        cur_limb,
        N
    )

    return res[0].reshape(-1, N), res[1].reshape(-1, N)


from ..functional import (  # noqa: E402,F401
    cv_add,
    cv_add_scalar,
    cv_automorphism_transform,
    cv_check,
    cv_drop_last_element_and_scale,
    cv_innerproduct,
    cv_keyswitch,
    cv_moddown,
    cv_modup,
    cv_mul,
    cv_mul_by_monomial,
    cv_mul_scalar,
    cv_neg,
    cv_sub,
    cv_sub_scalar,
    gen_scalar_tensor,
    cipher_fused_pairwise_mac,
)
