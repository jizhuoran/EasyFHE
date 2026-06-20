from __future__ import annotations

import math
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


def _native_plaintext_batch(plaintext):
    return _component_3d(plaintext.cv[0])


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
    return torch.neg_mod(x_component, x_component, modulus, cur_limbs=cur_limbs)


def cv_add(x, y, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    y_component = _component_3d(y)
    if inplace:
        torch.add_mod_(x_component, y_component, modulus, cur_limbs=cur_limbs)
        return x
    return torch.add_mod(x_component, y_component, modulus, cur_limbs=cur_limbs)


def cv_sub(x, y, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    y_component = _component_3d(y)
    if inplace:
        torch.sub_mod_(x_component, y_component, modulus, cur_limbs=cur_limbs)
        return x
    return torch.sub_mod(x_component, y_component, modulus, cur_limbs=cur_limbs)


def cv_mul(x, y, modulus, barret_mu, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    y_component = _component_3d(y)
    if inplace:
        torch.mul_mod_(x_component, y_component, modulus, barret_mu, cur_limbs=cur_limbs)
        return x
    return torch.mul_mod(x_component, y_component, modulus, barret_mu, cur_limbs=cur_limbs)


def cv_add_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    if inplace:
        torch.add_scalar_mod_(x_component, scalar, modulus, cur_limbs=cur_limbs)
        return x
    return torch.add_scalar_mod(x_component, scalar, modulus, cur_limbs=cur_limbs)


def cv_sub_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    if inplace:
        torch.sub_scalar_mod_(x_component, scalar, modulus, cur_limbs=cur_limbs)
        return x
    return torch.sub_scalar_mod(x_component, scalar, modulus, cur_limbs=cur_limbs)


def cv_mul_scalar(x, scalar, modulus, barret_mu, cur_limbs, inplace=False):
    x_component = _component_3d(x)
    if inplace:
        torch.mul_scalar_mod_(x_component, scalar, modulus, barret_mu, cur_limbs=cur_limbs)
        return x
    return torch.mul_scalar_mod(x_component, scalar, modulus, barret_mu, cur_limbs=cur_limbs)


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


def cv_innerproduct_broadcast(
    x: Tensor,
    curr_limbs: int,
    special_mod_starts: Tensor | list[int] | tuple[int, ...] | int,
    swk_bxs: list[Tensor],
    swk_axs: list[Tensor],
    context: Context,
) -> Tensor:
    x_3d = _component_3d(x)
    if isinstance(special_mod_starts, int):
        special_mod_starts = [special_mod_starts]
    if not torch.is_tensor(special_mod_starts):
        special_mod_starts = torch.from_numpy(np.array(special_mod_starts, dtype=np.int64)).to(context.device)
    return torch.innerproduct_broadcast(
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


def cv_innerproduct_pairwise(
    x: Tensor,
    curr_limbs: int,
    special_mod_starts: Tensor | list[int] | tuple[int, ...] | int,
    swk_bxs: list[Tensor],
    swk_axs: list[Tensor],
    context: Context,
) -> Tensor:
    x_3d = _component_3d(x)
    if isinstance(special_mod_starts, int):
        special_mod_starts = [special_mod_starts]
    if not torch.is_tensor(special_mod_starts):
        special_mod_starts = torch.from_numpy(np.array(special_mod_starts, dtype=np.int64)).to(context.device)
    return torch.innerproduct_pairwise(
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
    inner_product = cv_innerproduct_broadcast(
        modup_res,
        cur_limbs,
        [special_mod_start],
        [swk_bx],
        [swk_ax],
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


def cv_mul_by_monomial(
    input: Tensor,
    l: int,
    monomialDeg: int,
    context: Context,
) -> None:
    component = _component_3d(input).unsqueeze(0)
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


def cv_finalize_fast_rotation_ext(
    key_product_bx,
    key_product_ax,
    key_product_indices,
    c0,
    c1,
    precomp_maps,
    cur_limbs,
    active_limbs,
    context,
):
    return torch.finalize_fast_rotation_ext(
        key_product_bx,
        key_product_ax,
        key_product_indices,
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


def cv_double_hoist_giant_sum_ext(
    base_bx,
    base_ax,
    key_product_bx,
    key_product_ax,
    c0,
    precomp_maps,
    cur_limbs,
    active_limbs,
    context,
):
    return torch.double_hoist_giant_sum_ext(
        base_bx,
        base_ax,
        key_product_bx,
        key_product_ax,
        c0,
        precomp_maps,
        context.QplusP_map[cur_limbs],
        context.PModq,
        context.barret_ratio,
        context.barret_k,
        cur_limbs,
        active_limbs,
        context.N,
    )


def cv_finalize_fast_rotation_q(
    moddown_bx,
    moddown_ax,
    key_product_indices,
    c0,
    c1,
    precomp_maps,
    cur_limbs,
    context,
):
    return torch.finalize_fast_rotation_q(
        moddown_bx,
        moddown_ax,
        key_product_indices,
        c0,
        c1,
        precomp_maps,
        context.primes,
        cur_limbs,
        context.N,
    )


def cv_hmul_relin_rescale(
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
    return torch.hmul_relin_rescale(
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


def cv_rescale_one_level(
    input: Tensor,
    cur_limbs: int,
    l: int,
    context: Context,
) -> Tensor:
    component = _component_3d(input)
    rescale = torch.rescale_one_level(
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


def cv_pre_encode_stage1(input: Tensor, slots: int, context: Context) -> tuple[Tensor, float]:
    log_slots = int(math.log2(int(slots)))
    encoded, max_value = torch.fhe_pre_encode_stage1(
        input=input,
        slots=int(slots),
        M=context.M,
        rotGroup=context.encode_params_rotGroup,
        ksiPows=context.encode_params_ksiPows,
        bitrev=context.encode_bitrev_indices[log_slots],
    )
    return encoded, float(max_value.cpu().numpy())


def cv_fused_encode_batch(
    packed: Tensor,
    *,
    cur_limbs: int,
    slots: int,
    scaling_factor: float,
    context: Context,
) -> Tensor:
    return torch.fused_encode_batch(
        packed,
        int(cur_limbs),
        int(slots),
        context.N,
        context.M,
        float(scaling_factor),
        context.encode_params_rotGroup,
        context.encode_params_ksiPows,
        context.QplusP_map[cur_limbs],
        context.QmaxdiffplusPmaxdiff_map[cur_limbs],
        context.QbarretRatioplusPbarretRatio_map[cur_limbs],
        context.QbarretKplusPbarretK_map[cur_limbs],
        context.power_of_roots_shoup,
        context.power_of_roots,
    )


def cv_encrypt(ptx, pk0, pk1, l, logn, nh, moduli_p, moduli_q, context):
    cur_limbs = int(l)

    return torch.encrypt(
        ptx=ptx.contiguous(),
        pk0=pk0.contiguous(),
        pk1=pk1.contiguous(),
        l=cur_limbs,
        logn=logn,
        nh=nh,
        moduliP_scalar=moduli_p,
        moduliQ_scalar=moduli_q,
        primes=context.QplusP_map[cur_limbs],
        max_int_diffs=context.QmaxdiffplusPmaxdiff_map[cur_limbs],
        barret_ratio=context.QbarretRatioplusPbarretRatio_map[cur_limbs],
        barret_k=context.QbarretKplusPbarretK_map[cur_limbs],
        power_of_roots_shoup=context.power_of_roots_shoup,
        power_of_roots=context.power_of_roots,
    )


def cv_decrypt_decode(ct0, ct1, secret_key, moduli_q, roots_q, cur_limbs, plaintext_modulus_bits, noise_scale_deg, slots):
    if not hasattr(torch, "ckks_decrypt_decode"):
        return None
    return torch.ckks_decrypt_decode(
        ct0.cpu(),
        ct1.cpu(),
        secret_key.cpu(),
        moduli_q.cpu(),
        roots_q.cpu(),
        cur_limbs=int(cur_limbs),
        plaintext_modulus_bits=int(plaintext_modulus_bits),
        noise_scale_deg=int(noise_scale_deg),
        slots=int(slots),
    )


def cv_decode_phase_cuda(phase, moduli_q, crt_inv_moduli, cur_limbs, plaintext_modulus_bits, noise_scale_deg, slots, context):
    if not hasattr(torch, "ckks_decode_phase"):
        return None
    return torch.ckks_decode_phase(
        phase,
        moduli_q,
        crt_inv_moduli,
        context.inverse_power_of_roots_div_two,
        context.inverse_scaled_power_of_roots_div_two,
        context.encode_params_rotGroup,
        context.encode_params_ksiPows,
        cur_limbs=int(cur_limbs),
        plaintext_modulus_bits=int(plaintext_modulus_bits),
        noise_scale_deg=int(noise_scale_deg),
        slots=int(slots),
    )


def cipher_grouped_pairwise_mac(cipher, plaintext, groups, context):
    active_limbs = cipher.state.cur_limbs + (context.K if cipher.is_ext else 0)
    plaintext_values = _native_plaintext_batch(plaintext)
    res = torch.batched_pairwise_mac(
        _component_3d(cipher.cv[0]),
        _component_3d(cipher.cv[1]),
        plaintext_values,
        context.QplusP_map[cipher.state.cur_limbs],
        context.QbarretRatioplusPbarretRatio_map[cipher.state.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.state.cur_limbs],
        groups,
        cipher.batch_size,
        active_limbs,
        context.N,
    )
    return res


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
    if int(scalars.shape[2]) != int(cipher.state.cur_limbs):
        raise ValueError(
            "cipher_grouped_scalar_weighted_acc scalar limb mismatch: "
            f"{int(scalars.shape[2])} != {cipher.state.cur_limbs}"
        )
    res = torch.grouped_scalar_weighted_acc(
        _component_3d(cipher.cv[0]),
        _component_3d(cipher.cv[1]),
        scalars.contiguous(),
        context.moduliQ,
        context.QbarretRatioplusPbarretRatio_map[cipher.state.cur_limbs],
        context.QbarretKplusPbarretK_map[cipher.state.cur_limbs],
        groups,
        cipher.batch_size,
        cipher.state.cur_limbs,
        context.N,
        int(strategy),
    )
    return res
