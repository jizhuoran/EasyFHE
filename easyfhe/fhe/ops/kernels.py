from __future__ import annotations

from typing import TYPE_CHECKING

import easyfhe as torch
import numpy as np

if TYPE_CHECKING:
    from ..context import Context

Tensor = torch.Tensor


def _component_3d(t):
    if t.dim() != 3:
        raise ValueError(f"expected component tensor [batch, limbs, N], got shape {tuple(t.shape)}")
    return t


def _to_2d(t, n):
    if t.dim() == 1:
        return t.reshape(-1, n)
    return t


def _rhs_component(rhs, coeff_count):
    if rhs.dim() < 2:
        rhs = _to_2d(rhs, coeff_count)
    return _component_3d(rhs) if rhs.dim() >= 2 else rhs


def _preserve_limb_capacity(template, active):
    if not hasattr(template, "shape") or not hasattr(active, "shape"):
        return active
    if active.dim() < 2 or template.dim() < 2:
        return active
    if active.shape[-1] != template.shape[-1]:
        return active
    capacity = int(template.shape[-2])
    active_limbs = int(active.shape[-2])
    if active_limbs > capacity:
        return active
    desired_shape = tuple(active.shape[:-2]) + (capacity, active.shape[-1])
    if tuple(active.shape) == desired_shape:
        return active
    out = active.new_empty(desired_shape)
    out[..., :active_limbs, :] = active
    return out


def _native_result_like(template, active):
    return _preserve_limb_capacity(template, active)


def _native_plaintext_batch(plaintext):
    return _component_3d(plaintext.cv[0])


def native_op_available(name):
    return hasattr(torch, name)


def gen_scalar_tensor(scalar, modulus, cur_limbs):
    if isinstance(scalar, int):
        scalar_list = [int(int(scalar) % int(modulus[l])) for l in range(cur_limbs)]
    else:
        scalar_list = [int(int(scalar[l]) % int(modulus[l])) for l in range(cur_limbs)]
    return torch.from_numpy(np.array(scalar_list, dtype=np.uint64))


def cv_neg(x, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    if inplace:
        torch.neg_mod_(x_component, x_component, modulus, cur_limbs=cur_limbs)
        return x
    return _native_result_like(x, torch.neg_mod(x_component, x_component, modulus, cur_limbs=cur_limbs))


def cv_add(x, y, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    y_component = _rhs_component(y, x.shape[-1])
    if inplace:
        torch.add_mod_(x_component, y_component, modulus, cur_limbs=cur_limbs)
        return x
    return _native_result_like(x, torch.add_mod(x_component, y_component, modulus, cur_limbs=cur_limbs))


def cv_sub(x, y, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    y_component = _rhs_component(y, x.shape[-1])
    if inplace:
        torch.sub_mod_(x_component, y_component, modulus, cur_limbs=cur_limbs)
        return x
    return _native_result_like(x, torch.sub_mod(x_component, y_component, modulus, cur_limbs=cur_limbs))


def cv_mul(x, y, modulus, barret_mu, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    y_component = _rhs_component(y, x.shape[-1])
    if inplace:
        torch.mul_mod_(x_component, y_component, modulus, barret_mu, cur_limbs=cur_limbs)
        return x
    return _native_result_like(x, torch.mul_mod(x_component, y_component, modulus, barret_mu, cur_limbs=cur_limbs))


def cv_add_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    if inplace:
        torch.add_scalar_mod_(x_component, scalar, modulus, cur_limbs=cur_limbs)
        return x
    return _native_result_like(x, torch.add_scalar_mod(x_component, scalar, modulus, cur_limbs=cur_limbs))


def cv_sub_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    if inplace:
        torch.sub_scalar_mod_(x_component, scalar, modulus, cur_limbs=cur_limbs)
        return x
    return _native_result_like(x, torch.sub_scalar_mod(x_component, scalar, modulus, cur_limbs=cur_limbs))


def cv_mul_scalar(x, scalar, modulus, barret_mu, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    if inplace:
        torch.mul_scalar_mod_(x_component, scalar, modulus, barret_mu, cur_limbs=cur_limbs)
        return x
    return _native_result_like(x, torch.mul_scalar_mod(x_component, scalar, modulus, barret_mu, cur_limbs=cur_limbs))


def cv_add_pair(in0_c0, in0_c1, in1_c0, in1_c1, modulus, cur_limbs):
    return torch.cv_add_pair(
        in0_c0,
        in0_c1,
        in1_c0,
        in1_c1,
        modulus,
        cur_limbs=cur_limbs,
    )


def cv_add_pair_(in0_c0, in0_c1, in1_c0, in1_c1, modulus, cur_limbs):
    return torch.cv_add_pair_(
        in0_c0,
        in0_c1,
        in1_c0,
        in1_c1,
        modulus,
        cur_limbs=cur_limbs,
    )


def cv_sub_pair(in0_c0, in0_c1, in1_c0, in1_c1, modulus, cur_limbs):
    return torch.cv_sub_pair(
        in0_c0,
        in0_c1,
        in1_c0,
        in1_c1,
        modulus,
        cur_limbs=cur_limbs,
    )


def cv_sub_pair_(in0_c0, in0_c1, in1_c0, in1_c1, modulus, cur_limbs):
    return torch.cv_sub_pair_(
        in0_c0,
        in0_c1,
        in1_c0,
        in1_c1,
        modulus,
        cur_limbs=cur_limbs,
    )


def cv_mul_pt_pair(c0, c1, plaintext, modulus, barret_mu, cur_limbs):
    return torch.cv_mul_pt_pair(
        c0,
        c1,
        plaintext,
        modulus,
        barret_mu,
        cur_limbs=cur_limbs,
    )


def cv_mul_pt_pair_(c0, c1, plaintext, modulus, barret_mu, cur_limbs):
    return torch.cv_mul_pt_pair_(
        c0,
        c1,
        plaintext,
        modulus,
        barret_mu,
        cur_limbs=cur_limbs,
    )


def cv_mul_scalar_pair(c0, c1, scalar, modulus, barret_mu, cur_limbs):
    return torch.cv_mul_scalar_pair(
        c0,
        c1,
        scalar,
        modulus,
        barret_mu,
        cur_limbs=cur_limbs,
    )


def cv_mul_scalar_pair_(c0, c1, scalar, modulus, barret_mu, cur_limbs):
    return torch.cv_mul_scalar_pair_(
        c0,
        c1,
        scalar,
        modulus,
        barret_mu,
        cur_limbs=cur_limbs,
    )


def cv_modup(
    x: Tensor,
    curr_limbs: int,
    context: Context,
) -> Tensor:
    x_3d = _component_3d(x)
    beta = (curr_limbs + context.alpha - 1) // context.alpha
    return torch.modup(
        x_3d,
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
    )


def cv_moddown(
    x: Tensor,
    curr_limbs: int,
    context: Context,
) -> Tensor:
    x_3d = _component_3d(x)
    return torch.moddown(
        x_3d,
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


def cv_innerproduct(
    x: Tensor,
    curr_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
) -> Tensor:
    x_3d = _component_3d(x)
    return torch.innerproduct(
        x_3d,
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


def cv_innerproduct_broadcast_cipher_pair(
    x: Tensor,
    curr_limbs: int,
    special_mod_starts: Tensor,
    swk_bxs: list[Tensor],
    swk_axs: list[Tensor],
    context: Context,
) -> Tensor:
    x_3d = _component_3d(x)
    return torch.innerproduct_broadcast_cipher_pair(
        x_3d,
        bx=swk_bxs,
        ax=swk_axs,
        curr_limbs=curr_limbs,
        alpha=context.alpha,
        L=context.L,
        N=context.N,
        special_mod_start=special_mod_starts,
        primes=context.primes,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        workspace=context.inner_workspace,
    )


def cv_keyswitch(
    input: Tensor,
    cur_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
) -> Tensor:
    modup_res = cv_modup(
        input,
        curr_limbs=cur_limbs,
        context=context,
    )
    inner_product = cv_innerproduct(
        modup_res,
        cur_limbs,
        special_mod_start,
        swk_bx,
        swk_ax,
        context,
    )
    return tuple(cv_moddown(component, cur_limbs, context) for component in inner_product)


def cv_hrot(
    c0: Tensor,
    c1: Tensor,
    curr_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    inverse_precomp_map: Tensor,
    context: Context,
    add_bx: Tensor | None = None,
    add_ax: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    beta = (curr_limbs + context.alpha - 1) // context.alpha
    return torch.hrot(
        c0,
        c1,
        swk_bx,
        swk_ax,
        inverse_precomp_map,
        curr_limbs=curr_limbs,
        special_mod_start=special_mod_start,
        L=context.L,
        beta=beta,
        N=context.N,
        alpha=context.alpha,
        add_bx=add_bx,
        add_ax=add_ax,
        hat_inverse_vec_modup=context.hat_inverse_vec_modup,
        hat_inverse_vec_shoup_modup=context.hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j_modup=context.prod_q_i_mod_q_j_modup[curr_limbs - 1],
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
        inner_workspace=context.inner_workspace,
    )


def cv_automorphism_transform(
    input: Tensor,
    cur_limbs: int,
    i: int,
    context: Context,
) -> Tensor:
    if i < 0:
        raise ValueError("i should be non-negative")
    input_3d = _component_3d(input)
    return torch.automorphism_transform(
        input_3d, l=cur_limbs, N=context.N, precomp_vec=context.get_precompute_auto(i)
    )


def cv_mul_by_monomial(
    input: Tensor,
    l: int,
    monomialDeg: int,
    context: Context,
) -> None:
    component = _component_3d(input)
    torch.mul_by_monomial_(
        component,
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


def cv_fast_rotate_ext_batch_finalize_compact_pair(
    key_product_bx,
    key_product_ax,
    product_indices,
    c0,
    c1,
    precomp_maps,
    cur_limbs,
    active_limbs,
    context,
):
    return torch.fast_rotate_ext_batch_finalize_compact_pair(
        key_product_bx,
        key_product_ax,
        product_indices,
        c0,
        c1,
        precomp_maps,
        context.primes,
        context.PModq,
        context.barret_ratio,
        context.barret_k,
        cur_limbs,
        active_limbs,
        context.N,
    )


def cv_fast_rotate_batch_finalize_compact_pair(
    moddown_bx,
    moddown_ax,
    product_indices,
    c0,
    c1,
    precomp_maps,
    cur_limbs,
    context,
):
    return torch.fast_rotate_batch_finalize_compact_pair(
        moddown_bx,
        moddown_ax,
        product_indices,
        c0,
        c1,
        precomp_maps,
        context.primes,
        cur_limbs,
        context.N,
    )


def cv_hmul_double_rescale(
    c0: Tensor,
    c1: Tensor,
    d0: Tensor,
    d1: Tensor,
    curr_limbs: int,
    special_mod_start: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
    *,
    apply_double: bool = False,
    post_op: int = 0,
    post_c0: Tensor | None = None,
    post_c1: Tensor | None = None,
    post_scalar: Tensor | None = None,
) -> Tensor:
    beta = (curr_limbs + context.alpha - 1) // context.alpha
    return torch.hmul_double_rescale(
        c0.contiguous(),
        c1.contiguous(),
        d0.contiguous(),
        d1.contiguous(),
        swk_bx,
        swk_ax,
        curr_limbs=curr_limbs,
        special_mod_start=special_mod_start,
        L=context.L,
        beta=beta,
        N=context.N,
        alpha=context.alpha,
        old_prime=context.primes_list[curr_limbs - 1],
        primes=context.primes,
        q_mu=context.q_mu,
        barret_ratio=context.barret_ratio,
        barret_k=context.barret_k,
        hat_inverse_vec_modup=context.hat_inverse_vec_modup,
        hat_inverse_vec_shoup_modup=context.hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j_modup=context.prod_q_i_mod_q_j_modup[curr_limbs - 1],
        hat_inverse_vec_moddown=context.hat_inverse_vec_moddown,
        hat_inverse_vec_shoup_moddown=context.hat_inverse_vec_shoup_moddown,
        prod_q_i_mod_q_j_moddown=context.prod_q_i_mod_q_j_moddown,
        prod_inv_moddown=context.prod_inv_moddown,
        prod_inv_shoup_moddown=context.prod_inv_shoup_moddown,
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
        inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        switch_modulus_map=context.switch_modulus_map,
        qlql_inv_mod_ql_div_ql_mod_q=context.qlql_inv_mod_ql_div_ql_mod_q,
        qlql_inv_mod_ql_div_ql_mod_q_shoup=context.qlql_inv_mod_ql_div_ql_mod_q_shoup,
        q_inv_mod_q=context.q_inv_mod_q,
        q_inv_mod_q_shoup=context.q_inv_mod_q_shoup,
        inner_workspace=context.inner_workspace,
        apply_double=bool(apply_double),
        post_op=int(post_op),
        post_c0=None if post_c0 is None else post_c0.contiguous(),
        post_c1=None if post_c1 is None else post_c1.contiguous(),
        post_scalar=None if post_scalar is None else post_scalar.contiguous(),
    )


def cv_drop_last_element_and_scale(
    input: Tensor,
    cur_limbs: int,
    l: int,
    context: Context,
) -> Tensor:
    component = _component_3d(input)
    rescale = torch.drop_last_element_and_scale(
        component,
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
    return rescale


def cv_mod_raise(input: Tensor, L0: int, context: Context) -> Tensor:
    return torch.mod_raise(
        input,
        N=context.N,
        L0=L0,
        old_prime=context.primes_list[0],
        primes=context.primes,
        switch_modulus_map=context.switch_modulus_map,
        inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
    )


def cv_encode(input, N, cur_limbs, slots, scaling_factor, is_ext, context):
    return torch.encode(
        input=input,
        N=N,
        cur_limbs=cur_limbs,
        slots=slots,
        scaling_factor=scaling_factor,
        is_ext=is_ext,
        sizeP=context.primes.shape[0] - context.L,
        primes=context.QplusP_map[cur_limbs],
        max_int_diffs=context.QmaxdiffplusPmaxdiff_map[cur_limbs],
        barret_ratio=context.QbarretRatioplusPbarretRatio_map[cur_limbs],
        barret_k=context.QbarretKplusPbarretK_map[cur_limbs],
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
    )


def cv_encrypt(ptx, pk0, pk1, l, logn, nh, moduli_p, moduli_q, context):
    cur_limbs = int(l)

    def _cpu(value):
        return value.cpu() if torch.is_tensor(value) else value

    return torch.encrypt(
        ptx=ptx,
        pk0=pk0,
        pk1=pk1,
        l=cur_limbs,
        logn=logn,
        nh=nh,
        moduliP_scalar=moduli_p,
        moduliQ_scalar=moduli_q,
        primes=_cpu(context.QplusP_map[cur_limbs]),
        max_int_diffs=_cpu(context.QmaxdiffplusPmaxdiff_map[cur_limbs]),
        barret_ratio=_cpu(context.QbarretRatioplusPbarretRatio_map[cur_limbs]),
        barret_k=_cpu(context.QbarretKplusPbarretK_map[cur_limbs]),
        power_of_roots_shoup=_cpu(context.power_of_roots_shoup),
        power_of_roots=_cpu(context.power_of_roots),
    )


def cipher_fused_grouped_pairwise_mac(cipher, plaintext, groups, context):
    active_limbs = cipher.cur_limbs + (context.K if cipher.is_ext else 0)
    plaintext_values = _native_plaintext_batch(plaintext)
    res = torch.batched_pairwise_mac(
        _component_3d(cipher.cv[0]),
        _component_3d(cipher.cv[1]),
        plaintext_values,
        context.QplusP_map[cipher.cur_limbs],
        context.QbarretRatioplusPbarretRatio_map[cipher.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.cur_limbs],
        groups,
        cipher.batch_size,
        active_limbs,
        context.N,
    )
    return (
        _preserve_limb_capacity(cipher.cv[0], res[0]),
        _preserve_limb_capacity(cipher.cv[1], res[1]),
    )


def cipher_fused_broadcast_mac(cipher, plaintext, context):
    active_limbs = cipher.cur_limbs + (context.K if cipher.is_ext else 0)
    plaintext_values = _native_plaintext_batch(plaintext)
    res = torch.fused_broadcast_mac(
        _component_3d(cipher.cv[0]),
        _component_3d(cipher.cv[1]),
        plaintext_values,
        context.QplusP_map[cipher.cur_limbs],
        context.QbarretRatioplusPbarretRatio_map[cipher.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.cur_limbs],
        plaintext.batch_size,
        active_limbs,
        context.N,
    )
    return (
        _preserve_limb_capacity(cipher.cv[0], res[0]),
        _preserve_limb_capacity(cipher.cv[1], res[1]),
    )


def cipher_scalar_weighted_acc(cipher, scalars, context):
    if cipher.is_ext:
        raise ValueError("cipher_scalar_weighted_acc expects a non-ext cipher batch")
    if int(scalars.shape[0]) != int(cipher.batch_size):
        raise ValueError(
            "cipher_scalar_weighted_acc scalar batch mismatch: "
            f"{int(scalars.shape[0])} != {cipher.batch_size}"
        )
    if int(scalars.shape[1]) != int(cipher.cur_limbs):
        raise ValueError(
            "cipher_scalar_weighted_acc scalar limb mismatch: "
            f"{int(scalars.shape[1])} != {cipher.cur_limbs}"
        )
    res = torch.scalar_weighted_acc(
        _component_3d(cipher.cv[0]),
        _component_3d(cipher.cv[1]),
        scalars,
        context.moduliQ,
        context.QbarretRatioplusPbarretRatio_map[cipher.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.cur_limbs],
        cipher.batch_size,
        cipher.cur_limbs,
        context.N,
    )
    return (
        _preserve_limb_capacity(cipher.cv[0], res[0]),
        _preserve_limb_capacity(cipher.cv[1], res[1]),
    )


def cipher_grouped_scalar_weighted_acc(cipher, scalars, context, *, strategy=-1):
    if cipher.is_ext:
        raise ValueError("cipher_grouped_scalar_weighted_acc expects a non-ext cipher batch")
    if scalars.dim() != 3:
        raise ValueError(
            "cipher_grouped_scalar_weighted_acc expects scalars with shape "
            "[groups, degree, limbs]"
        )
    groups = int(scalars.shape[0])
    degree = int(scalars.shape[1])
    if degree != int(cipher.batch_size):
        raise ValueError(
            "cipher_grouped_scalar_weighted_acc scalar degree mismatch: "
            f"{degree} != {cipher.batch_size}"
        )
    if int(scalars.shape[2]) != int(cipher.cur_limbs):
        raise ValueError(
            "cipher_grouped_scalar_weighted_acc scalar limb mismatch: "
            f"{int(scalars.shape[2])} != {cipher.cur_limbs}"
        )
    res = torch.grouped_scalar_weighted_acc(
        _component_3d(cipher.cv[0]),
        _component_3d(cipher.cv[1]),
        scalars.contiguous(),
        context.moduliQ,
        context.QbarretRatioplusPbarretRatio_map[cipher.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.cur_limbs],
        groups,
        cipher.batch_size,
        cipher.cur_limbs,
        context.N,
        int(strategy),
    )
    return (
        _preserve_limb_capacity(cipher.cv[0], res[0]),
        _preserve_limb_capacity(cipher.cv[1], res[1]),
    )
