import torch
from .context import Context
import numpy as np

Tensor = torch.Tensor


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


def gen_scalar_tensor(scalar, modulus, cur_limbs):
    if isinstance(scalar, int):
        scalar_list = [int(int(scalar) % int(modulus[l])) for l in range(cur_limbs)]
    else:
        scalar_list = [int(int(scalar[l]) % int(modulus[l])) for l in range(cur_limbs)]
    return torch.from_numpy(np.array(scalar_list, dtype=np.uint64)).cuda()


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
    return torch.modup(
        x,
        curr_limbs=curr_limbs,
        L=context.L,
        hat_inverse_vec=context.hat_inverse_vec_modup,
        hat_inverse_vec_shoup=context.hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j=context.prod_q_i_mod_q_j_modup[curr_limbs - 1],
        primes=context.primes,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        degree=context.N,
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


def cv_innerproduct(
    x: Tensor, curr_limbs: int, swk_bx: Tensor, swk_ax: Tensor, context: Context
) -> Tensor:
    x.reshape(-1)
    res = torch.innerproduct(
        context.inner_out,
        x,
        bx=swk_bx,
        ax=swk_ax,
        curr_limbs=curr_limbs,
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
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
) -> list:
    modup_res = cv_modup(input, curr_limbs=cur_limbs, context=context)
    inner_product = cv_innerproduct(
        modup_res.reshape(-1), cur_limbs, swk_bx, swk_ax, context
    )

    moddown_bx = cv_moddown(inner_product[0], curr_limbs=cur_limbs, context=context)

    moddown_ax = cv_moddown(inner_product[1], curr_limbs=cur_limbs, context=context)

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
    input: Tensor, cur_limbs: int, i: int, context: Context
) -> Tensor:
    if i < 0:
        raise ValueError("rotation index must be non-negative")
    return torch.automorphism_transform(
        input, l=cur_limbs, N=context.N, precomp_vec=context.precompute_auto_map[i]
    )


def cv_mul_by_monomial(
    input: Tensor,
    l: int,
    monomialDeg: int,
    context: Context,
) -> None:
    torch.mul_by_monomial_(
        input,
        primes=context.primes,
        l=l,
        N=context.N,
        M=context.M,
        monomialDeg=monomialDeg,
        L=context.L,
        inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
    )
