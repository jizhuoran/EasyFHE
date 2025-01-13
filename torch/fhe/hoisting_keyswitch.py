import numpy as np
from .ciphertext import Cipher
from . import functional as F
from .utils import check_meta_equal
import torch

def eval_fast_rotation_precompute(input, curr_limbs, cryptoContext):
    return F.cv_modup(input, curr_limbs, cryptoContext)

def eval_fast_key_switch_core_ext(d2Tilde, auto_index, curr_limbs, cryptoContext):
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    res = F.cv_innerproduct(
        d2Tilde.reshape(-1),
        curr_limbs=curr_limbs,
        context=cryptoContext,
        swk_bx=swk[0],
        swk_ax=swk[1]
    )
    return res[0], res[1]

def key_switch_down(cipher, cryptoContext):
    res_bx = F.cv_moddown(cipher.cv[0], cipher.cur_limbs, cryptoContext)
    res_ax = F.cv_moddown(cipher.cv[1], cipher.cur_limbs, cryptoContext)
    return Cipher([res_bx, res_ax], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)

def key_switch_ext(cipher, add_first, cryptoContext):
    # cv0 = torch.zeros(((cipher.cur_limbs + cryptoContext.K) << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
    # cv1 = torch.zeros(((cipher.cur_limbs + cryptoContext.K) << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
    # if add_first:
    #     cv0[:cipher.cur_limbs, :] = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
    #                                         cryptoContext.q_mu_cuda, cipher.cur_limbs)

    # cv1[:cipher.cur_limbs, :] = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
    #                                     cryptoContext.q_mu_cuda, cipher.cur_limbs)
    # return Cipher([cv0, cv1], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)
    
    assert add_first == True
    cv0 = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)
    cv1 = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)

    cv0 = torch.cat((cv0, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)), dim=0)
    cv1 = torch.cat((cv1, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)), dim=0)

    return Cipher([cv0, cv1], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)

@check_meta_equal
def eval_add_ext(cipher0, cipher1, cryptoContext):
    moduli = cryptoContext.BsContext.QplusP_map[cipher0.cur_limbs]
    cv = [
        F.cv_add(cv0, cv1, moduli, cipher0.cur_limbs + cryptoContext.K)
        for cv0, cv1 in zip(cipher0.cv, cipher1.cv)
    ]
    return Cipher(cv, cipher0.cur_limbs, cipher0.scaling_factor, cipher0.noise_deg, cipher0.slots)

def cv_add_ext(in0, in1, cur_limbs, cryptoContext):
    moduli = cryptoContext.BsContext.QplusP_map[cur_limbs]
    res = F.cv_add(in0, in1, moduli, in0.shape[0])
    return res

#todo: it is ct*pt in extent form, refactor?
def eval_mult_ext(cipher, pt, cryptoContext):
    moduli = cryptoContext.BsContext.QplusP_map[cipher.cur_limbs]
    mu = cryptoContext.BsContext.QmuplusPmu_map[cipher.cur_limbs]
    cv0 = F.cv_mul(cipher.cv[0], pt.mx, moduli, mu, cipher.cur_limbs + cryptoContext.K)
    cv1 = F.cv_mul(cipher.cv[1], pt.mx, moduli, mu, cipher.cur_limbs + cryptoContext.K)
    return Cipher([cv0, cv1], cipher.cur_limbs, cipher.scaling_factor*pt.scaling_factor, cipher.noise_deg+pt.noise_deg, cipher.slots)


def eval_fast_rotation(cipher, index, digits, cryptoContext):
    if index == 0: return cipher.clone()

    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.find_auto_index(index)

    # EvalFastKeySwitchCore = InnerProduct + ModDown
    sumbxmult, sumaxmult = eval_fast_key_switch_core_ext(digits, auto_index, cipher.cur_limbs, cryptoContext)
    cv0 = F.cv_moddown(sumbxmult, cipher.cur_limbs, cryptoContext)
    cv1 = F.cv_moddown(sumaxmult, cipher.cur_limbs, cryptoContext)

    # post add after ks
    cv0 = F.cv_add(cipher.cv[0], cv0, cryptoContext.moduliQ_cuda, cipher.cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    cv0 = F.cv_automorphism_transform(cv0, cipher.cur_limbs, auto_index, cryptoContext)
    cv1 = F.cv_automorphism_transform(cv1, cipher.cur_limbs, auto_index, cryptoContext)

    return Cipher([cv0, cv1], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)

def eval_fast_rotation_ext(cipher, digits, index, add_first, cryptoContext):
    
    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.find_auto_index(index)

    # Inner Product
    sumbxmult, sumaxmult = eval_fast_key_switch_core_ext(digits, auto_index, cipher.cur_limbs, cryptoContext)

    if (add_first):
        cMult = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                cryptoContext.q_mu_cuda, cipher.cur_limbs)
        sumbxmult = F.cv_add(sumbxmult, cMult, cryptoContext.moduliQ_cuda, cipher.cur_limbs, inplace=True)

    cv0 = F.cv_automorphism_transform(sumbxmult, cipher.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
    cv1 = F.cv_automorphism_transform(sumaxmult, cipher.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
    return Cipher([cv0, cv1], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)




