import torch
from .context import Context
import numpy as np

Tensor = torch.Tensor

# def cpp_cuda_adaptor(func):
#     def wrapper(*args, **kwargs):
#         args_new = []
#         for i in range(len(args)):
#             if isinstance(args[i], torch.Tensor):
#                 args_new.append(args[i].cpu())
#             elif isinstance(args[i], Context):
#                 args_new.append(args[i])
#                 args_new[-1].cpu()
#             else:
#                 args_new.append(args[i])
#         result = func(*args_new, **kwargs)
#         for i in range(len(args_new)):
#             if isinstance(args_new[i], Context):
#                 args_new[i].cuda()
#         if isinstance(result, list):
#             return [r.cuda() for r in result]
#         elif isinstance(result, torch.Tensor):
#             result = result.cuda()
#         return result
#     return wrapper

# def cpp_cuda_compare(func):
#     def wrapper(*args, **kwargs):
#         args_new = []
#         for i in range(len(args)):
#             print("type of args[i]: ", type(args[i]))
#             if isinstance(args[i], torch.Tensor):
#                 args_new.append(args[i].clone().cpu())
#             elif isinstance(args[i], Context):
#                 args_new.append(args[i])
#                 args_new[-1].cpu()
#             else:
#                 print("type of args[i]: ", type(args[i]))
#                 args_new.append(args[i])
#         result = func(*args_new, **kwargs)
#         result_gt = func(*args, **kwargs)
#         for i in range(len(args_new)):
#             if isinstance(args_new[i], Context):
#                 args_new[i].cuda()

#         print("device of result: ", result.device)
#         print("device of result_gt: ", result_gt.device)
#         if not torch.allclose(result_gt.cpu(), result):
#             print("Test Failed")
#             print("Expected: ", result_gt)
#             print("Got: ", result)
#             exit(1)
#         else:
#             print("Test Passed")
#         return result_gt
#     return wrapper



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
    )

# does not copy back when CPU and GPU
def cv_set_zero(x, length):
    return torch.set_zero(x, length)

# @cpp_cuda_adaptor
def cv_neg(x, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.neg_mod_(x, x, modulus, cur_limbs=cur_limbs)
    else:
        return torch.neg_mod(x, x, modulus, cur_limbs=cur_limbs)

# @cpp_cuda_adaptor
def cv_add(x, y, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.add_mod_(x, y, modulus, cur_limbs=cur_limbs)
    else:
        return torch.add_mod(x, y, modulus, cur_limbs=cur_limbs)

# @cpp_cuda_adaptor
def cv_sub(x, y, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.sub_mod_(x, y, modulus, cur_limbs=cur_limbs)
    else:
        return torch.sub_mod(x, y, modulus, cur_limbs=cur_limbs)

# @cpp_cuda_adaptor
def cv_mul(x, y, modulus, barret_mu, cur_limbs, inplace=False):
    if inplace:
        return torch.mul_mod_(x, y, modulus, barret_mu, cur_limbs=cur_limbs)
    else:
        return torch.mul_mod(x, y, modulus, barret_mu, cur_limbs=cur_limbs)

# @cpp_cuda_adaptor
def cv_add_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.add_scalar_mod_(x, scalar, modulus, cur_limbs=cur_limbs)
    else:
        return torch.add_scalar_mod(x, scalar, modulus, cur_limbs=cur_limbs)

# @cpp_cuda_adaptor
def cv_sub_scalar(x, scalar, modulus, cur_limbs, inplace=False):
    if inplace:
        return torch.sub_scalar_mod_(x, scalar, modulus, cur_limbs=cur_limbs)
    else:
        return torch.sub_scalar_mod(x, scalar, modulus, cur_limbs=cur_limbs)

# @cpp_cuda_adaptor
def cv_mul_scalar(x, scalar, modulus, barret_mu, cur_limbs, inplace=False):
    if inplace:
        return torch.mul_scalar_mod_(x, scalar, modulus, barret_mu, cur_limbs=cur_limbs)
    else:
        return torch.mul_scalar_mod(x, scalar, modulus, barret_mu, cur_limbs=cur_limbs)

# @cpp_cuda_adaptor
def cv_modup(
    x: Tensor,
    curr_limbs: int,
    context: Context,
    inplace: bool = False,
) -> Tensor:
    beta = (curr_limbs + context.K - 1) // context.K
    if inplace:
        res = torch.modup_(
            context.modup_out,
            x,
            curr_limbs=curr_limbs,
            level=context.L,
            hat_inverse_vec=context.hat_inverse_vec_modup,
            hat_inverse_vec_shoup=context.hat_inverse_vec_shoup_modup,
            prod_q_i_mod_q_j=context.prod_q_i_mod_q_j_modup[curr_limbs - 1],
            primes=context.primes,
            barret_ratio=context.barret_ratio,
            barret_k=context.barret_k,
            beta=beta,
            degree=context.N,
            alpha=context.K,
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
            level=context.L,
            hat_inverse_vec=context.hat_inverse_vec_modup,
            hat_inverse_vec_shoup=context.hat_inverse_vec_shoup_modup,
            prod_q_i_mod_q_j=context.prod_q_i_mod_q_j_modup[curr_limbs - 1],
            primes=context.primes,
            barret_ratio=context.barret_ratio,
            barret_k=context.barret_k,
            beta=beta,
            degree=context.N,
            alpha=context.K,
            param_power_of_roots_shoup=context.power_of_roots_shoup,
            param_power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        )

    return res.reshape(-1, context.N)

# @cpp_cuda_adaptor
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
            level=context.L,
            alpha=context.K,
            param_degree=context.N,
            param_log_degree=context.logN,
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
            level=context.L,
            alpha=context.K,
            param_degree=context.N,
            param_log_degree=context.logN,
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

    return res.reshape(-1, context.N)

# @cpp_cuda_adaptor
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

# @cpp_cuda_adaptor
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

# @cpp_cuda_adaptor
def cv_innerproduct(
    x: Tensor,
    curr_limbs: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        res = torch.innerproduct_(context.inner_out, x, bx=swk_bx, ax=swk_ax, curr_limbs=curr_limbs,
                                  alpha=context.K, level=context.L, param_degree=context.N,
                                  primes=context.primes, barret_ratio=context.barret_ratio,
                                  barret_k=context.barret_k, workspace=context.inner_workspace)
    else:
        res = torch.innerproduct(context.inner_out, x, bx=swk_bx, ax=swk_ax, curr_limbs=curr_limbs,
                                 alpha=context.K, level=context.L, param_degree=context.N,
                                 primes=context.primes, barret_ratio=context.barret_ratio,
                                 barret_k=context.barret_k, workspace=context.inner_workspace)
    return res.reshape(2, -1, context.N)

def cv_keyswitch(
    input: Tensor,
    cur_limbs: int,
    swk_bx: Tensor,
    swk_ax: Tensor,
    context: Context,
    inplace: bool = False,
) -> Tensor:
    true_beta = int((cur_limbs + (context.K - 1)) / context.K)
    context.beta = true_beta
    modup_res = cv_modup(
        input,
        cur_limbs,
        context,
        inplace,
    )
    inner_product = cv_innerproduct(
        modup_res.reshape(-1),
        cur_limbs,
        swk_bx,
        swk_ax,
        context,
        inplace,
    )

    sumMult_bx = inner_product[0]
    sumMult_ax = inner_product[1]

    moddown_bx = cv_moddown(
        sumMult_bx,
        cur_limbs,
        context,
        False,
    )

    moddown_ax = cv_moddown(
        sumMult_ax,
        cur_limbs,
        context,
        False,
    )

    return [moddown_bx, moddown_ax]

# @cpp_cuda_adaptor
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
            level=context.L,
            param_degree=context.N,
            param_primes=context.primes,
            param_barret_ratio=context.barret_ratio,
            param_barret_k=context.barret_k,
            param_power_of_roots_shoup=context.power_of_roots_shoup,
            param_power_of_roots=context.power_of_roots,
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
            level=context.L,
            param_degree=context.N,
            param_primes=context.primes,
            param_barret_ratio=context.barret_ratio,
            param_barret_k=context.barret_k,
            param_power_of_roots_shoup=context.power_of_roots_shoup,
            param_power_of_roots=context.power_of_roots,
            inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
            qlql_inv_mod_ql_div_ql_mod_q=context.qlql_inv_mod_ql_div_ql_mod_q,
            qlql_inv_mod_ql_div_ql_mod_q_shoup=context.qlql_inv_mod_ql_div_ql_mod_q_shoup,
            q_inv_mod_q=context.q_inv_mod_q,
            q_inv_mod_q_shoup=context.q_inv_mod_q_shoup,
        )

    return rescale.reshape(-1, context.N)

# @cpp_cuda_adaptor
def cv_automorphism_transform(
    input: Tensor,
    l: int,
    i: int,
    context: Context
) -> Tensor:
    automorphism_transform = torch.automorphism_transform(
        context.automorphism_transform_out,
        input,
        l=int(l),
        N=context.N,
        i=int(i),
        precomp_vec=context.precompute_auto_map[i]
    )

    return automorphism_transform.reshape(-1, context.N)

# @cpp_cuda_adaptor
def cv_switch_modulus_with_intt_ntt(
    input: Tensor, 
    L0 : int,
    context: Context
    ) -> Tensor:
    switch_modulus = torch.switch_modulus(
        context.switch_modulus_out,
        input,
        primes = context.primes,
        N = context.N,
        L0 = L0,
        logN = context.logN,
        level = context.L,
        inverse_power_of_roots_div_two = context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two  = context.inverse_scaled_power_of_roots_div_two,
        param_power_of_roots_shoup = context.power_of_roots_shoup,
        param_power_of_roots = context.power_of_roots
    )
    return switch_modulus.reshape(-1, context.N)

# @cpp_cuda_compare
def cv_mul_by_monomial(
    input: Tensor,
    l: int,
    monomialDeg: int,
    context: Context,
) -> Tensor:
    mul_by_monomial = torch.mul_by_monomial(
        input,
        primes = context.primes,
        l = l,
        N = context.N,
        M = context.M,
        monomialDeg=monomialDeg,
        level=context.L,
        inverse_power_of_roots_div_two=context.inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two=context.inverse_scaled_power_of_roots_div_two,
        param_power_of_roots_shoup=context.power_of_roots_shoup,
        param_power_of_roots=context.power_of_roots
    )
    return mul_by_monomial.reshape(-1, context.N)
