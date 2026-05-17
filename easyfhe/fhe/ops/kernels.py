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


def _to_4d(t):
    if t.dim() == 2:
        return t.reshape(1, 1, t.shape[0], t.shape[1])
    if t.dim() == 3:
        return t.reshape(1, t.shape[0], t.shape[1], t.shape[2])
    return t


def _from_4d(t, orig_shape):
    if len(orig_shape) == 2:
        return t.reshape(-1, orig_shape[-1])
    if len(orig_shape) == 3:
        return t.reshape(orig_shape[0], -1, orig_shape[-1])
    return t


def _to_2d(t, n):
    if t.dim() == 1:
        return t.reshape(-1, n)
    return t


def cv_neg(x, modulus, cur_limbs, inplace=False):
    orig_shape = x.shape
    x_4d = _to_4d(x)
    if inplace:
        torch.neg_mod_(x_4d, x_4d, modulus, cur_limbs=cur_limbs)
        return x
    res = torch.neg_mod(x_4d, x_4d, modulus, cur_limbs=cur_limbs)
    return _from_4d(res, orig_shape)


def cv_add(x, y, modulus, cur_limbs, inplace=False):
    orig_shape = x.shape
    x_4d = _to_4d(x)
    if y.dim() < 2:
        y = _to_2d(y, x.shape[-1])
    y_4d = _to_4d(y) if y.dim() >= 2 else y
    if inplace:
        torch.add_mod_(x_4d, y_4d, modulus, cur_limbs=cur_limbs)
        return x
    res = torch.add_mod(x_4d, y_4d, modulus, cur_limbs=cur_limbs)
    return _from_4d(res, orig_shape)


def cv_sub(x, y, modulus, cur_limbs, inplace=False):
    orig_shape = x.shape
    x_4d = _to_4d(x)
    if y.dim() < 2:
        y = _to_2d(y, x.shape[-1])
    y_4d = _to_4d(y) if y.dim() >= 2 else y
    if inplace:
        torch.sub_mod_(x_4d, y_4d, modulus, cur_limbs=cur_limbs)
        return x
    res = torch.sub_mod(x_4d, y_4d, modulus, cur_limbs=cur_limbs)
    return _from_4d(res, orig_shape)


def cv_mul(x, y, modulus, barret_mu, cur_limbs, inplace=False):
    orig_shape = x.shape
    x_4d = _to_4d(x)
    if y.dim() < 2:
        y = _to_2d(y, x.shape[-1])
    y_4d = _to_4d(y) if y.dim() >= 2 else y
    if inplace:
        torch.mul_mod_(x_4d, y_4d, modulus, barret_mu, cur_limbs=cur_limbs)
        return x
    res = torch.mul_mod(x_4d, y_4d, modulus, barret_mu, cur_limbs=cur_limbs)
    return _from_4d(res, orig_shape)


def cv_add_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    orig_shape = x.shape
    x_4d = _to_4d(x)
    if inplace:
        torch.add_scalar_mod_(x_4d, scalar, modulus, cur_limbs=cur_limbs)
        return x
    res = torch.add_scalar_mod(x_4d, scalar, modulus, cur_limbs=cur_limbs)
    return _from_4d(res, orig_shape)


def cv_sub_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    orig_shape = x.shape
    x_4d = _to_4d(x)
    if inplace:
        torch.sub_scalar_mod_(x_4d, scalar, modulus, cur_limbs=cur_limbs)
        return x
    res = torch.sub_scalar_mod(x_4d, scalar, modulus, cur_limbs=cur_limbs)
    return _from_4d(res, orig_shape)


def cv_mul_scalar(x, scalar, modulus, barret_mu, cur_limbs, inplace=False):
    orig_shape = x.shape
    x_4d = _to_4d(x)
    if inplace:
        torch.mul_scalar_mod_(x_4d, scalar, modulus, barret_mu, cur_limbs=cur_limbs)
        return x
    res = torch.mul_scalar_mod(x_4d, scalar, modulus, barret_mu, cur_limbs=cur_limbs)
    return _from_4d(res, orig_shape)


def cv_modup(
    x: Tensor,
    curr_limbs: int,
    context: Context,
) -> Tensor:
    x_4d = _to_4d(x)
    beta = (curr_limbs + context.alpha - 1) // context.alpha
    return torch.modup(
        x_4d,
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
    x_4d = _to_4d(x)
    return torch.moddown(
        x_4d,
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


def cv_moddown_write(out: Tensor, x: Tensor, curr_limbs: int, context: Context) -> Tensor:
    x_4d = _to_4d(x)
    return torch.moddown_write(
        out,
        x_4d,
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
    )


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
    if x.dim() == 1:
        x_4d = x.reshape(1, 1, -1, context.N)
    else:
        x_4d = _to_4d(x)
    res = torch.innerproduct(
        x_4d,
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


def cv_innerproduct_write(
    out: Tensor,
    x: Tensor,
    curr_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
) -> Tensor:
    if x.dim() == 1:
        x_4d = x.reshape(1, 1, -1, context.N)
    else:
        x_4d = _to_4d(x)
    return torch.innerproduct_write(
        out,
        x_4d,
        bx=swk_bx,
        ax=swk_ax,
        curr_limbs=curr_limbs,
        alpha=context.alpha,
        special_mod_start=special_mod_start,
        L=context.L,
        N=context.N,
        primes=context.primes,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        workspace=context.inner_workspace,
    )


def cv_innerproduct_write_pair(
    out_bx: Tensor,
    out_ax: Tensor,
    x: Tensor,
    curr_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
) -> tuple[Tensor, Tensor]:
    if x.dim() == 1:
        x_4d = x.reshape(1, 1, -1, context.N)
    else:
        x_4d = _to_4d(x)
    res = torch.innerproduct_write_pair(
        out_bx,
        out_ax,
        x_4d,
        bx=swk_bx,
        ax=swk_ax,
        curr_limbs=curr_limbs,
        alpha=context.alpha,
        special_mod_start=special_mod_start,
        L=context.L,
        N=context.N,
        primes=context.primes,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        workspace=context.inner_workspace,
    )
    return res[0], res[1]


def cv_fast_rotate_ext_batch_finalize(key_products, pc0, pc1, precomp_maps, offsets, cur_limbs, context):
    active_limbs = key_products.shape[2]
    return torch.fast_rotate_ext_batch_finalize(
        key_products,
        pc0,
        pc1,
        precomp_maps,
        offsets,
        context.primes,
        cur_limbs,
        active_limbs,
        context.N,
    )


def cv_fast_rotate_ext_batch_finalize_pair(
    key_product_bx,
    key_product_ax,
    pc0,
    pc1,
    precomp_maps,
    offsets,
    cur_limbs,
    context,
):
    active_limbs = key_product_bx.shape[1]
    res = torch.fast_rotate_ext_batch_finalize_pair(
        key_product_bx,
        key_product_ax,
        pc0,
        pc1,
        precomp_maps,
        offsets,
        context.primes,
        cur_limbs,
        active_limbs,
        context.N,
    )
    return res[0], res[1]


def cv_fast_rotate_batch_finalize(moddown_products, c0, c1, precomp_maps, offsets, context):
    return torch.fast_rotate_batch_finalize(
        moddown_products,
        c0,
        c1,
        precomp_maps,
        offsets,
        context.primes,
        moddown_products.shape[2],
        context.N,
    )


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
        modup_res,
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
) -> Tensor:
    input_4d = _to_4d(input)
    rescale = torch.drop_last_element_and_scale(
        input_4d,
        curr_limbs=cur_limbs,
        l=l,
        L=context.L,
        N=context.N,
        old_prime=context.primes_list[cur_limbs - 1],
        primes=context.primes,
        switch_modulus_map=context.switch_modulus_map,
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
    input_4d = _to_4d(input)
    return torch.automorphism_transform(
        input_4d, l=cur_limbs, N=context.N, precomp_vec=context.get_precompute_auto(i)
    ).reshape(input.shape)


def cv_mul_by_monomial(
    input: Tensor,
    l: int,
    monomialDeg: int,
    context: Context,
) -> None:
    input_4d = _to_4d(input)
    torch.mul_by_monomial_(
        input_4d,
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

def _native_plaintext_batch(plaintext):
    values = plaintext.cv[0]
    if values.dim() == 3:
        values = values.unsqueeze(1)
    return values


def cipher_fused_grouped_pairwise_mac(cipher, plaintext, groups, context):
    active_limbs = cipher.cur_limbs + (context.K if cipher.is_ext else 0)
    cipher_values = torch.stack(cipher.cv, dim=0)
    plaintext_values = _native_plaintext_batch(plaintext)
    if _grouped_pairwise_mac_needs_cuda_fallback(cipher_values, cipher.batch_size, groups):
        return _cipher_fused_grouped_pairwise_mac_loop(
            cipher,
            plaintext_values,
            groups,
            active_limbs,
            context,
        )
    res = torch.batched_pairwise_mac(
        cipher_values,
        plaintext_values,
        context.QplusP_map[cipher.cur_limbs],
        context.QbarretRatioplusPbarretRatio_map[cipher.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.cur_limbs],
        groups,
        cipher.batch_size,
        active_limbs,
        context.N,
    )
    return res[0], res[1]


def _grouped_pairwise_mac_needs_cuda_fallback(cipher_values, cipher_batch_size, groups):
    if not cipher_values.is_cuda or int(groups) <= 1:
        return False
    block_size = 256
    bytes_per_uint64 = 8
    shared_bytes = int(cipher_batch_size) * block_size * bytes_per_uint64 * 2
    return shared_bytes > 48 * 1024


def _cipher_fused_grouped_pairwise_mac_loop(cipher, plaintext_values, groups, active_limbs, context):
    cipher_values = torch.stack(cipher.cv, dim=0)
    outputs = []
    for group in range(int(groups)):
        start = group * cipher.batch_size
        stop = start + cipher.batch_size
        outputs.append(
            torch.batched_pairwise_mac(
                cipher_values,
                plaintext_values[start:stop].contiguous(),
                context.QplusP_map[cipher.cur_limbs],
                context.QbarretRatioplusPbarretRatio_map[cipher.cur_limbs],
                context.QbarretKplusPbarretK_map[cipher.cur_limbs],
                1,
                cipher.batch_size,
                active_limbs,
                context.N,
            )[:, 0]
        )
    res = torch.stack(outputs, dim=1)
    return res[0], res[1]


def cipher_fused_pairwise_mac(cipher, plaintext, context):
    res = cipher_fused_grouped_pairwise_mac(cipher, plaintext, 1, context)
    return res[0][0], res[1][0]


def cipher_fused_broadcast_mac(cipher, plaintext, context):
    active_limbs = cipher.cur_limbs + (context.K if cipher.is_ext else 0)
    cipher_values = torch.stack(cipher.cv, dim=0)
    plaintext_values = _native_plaintext_batch(plaintext)
    res = torch.fused_broadcast_mac(
        cipher_values,
        plaintext_values,
        context.QplusP_map[cipher.cur_limbs],
        context.QbarretRatioplusPbarretRatio_map[cipher.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.cur_limbs],
        plaintext.batch_size,
        active_limbs,
        context.N,
    )
    return res[0], res[1]
