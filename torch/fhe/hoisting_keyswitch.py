import numpy as np
from .ciphertext import Cipher
from . import functional as F
from .utils import check_meta_equal, call_counter
import torch

def eval_rotation_ext_precompute(input, curr_limbs, cryptoContext):
    return F.cv_modup(input, curr_limbs, cryptoContext)



def key_switch_down(cipher, cryptoContext):
    res_bx = F.cv_moddown(cipher.cv[0], cipher.cur_limbs, cryptoContext)
    res_ax = F.cv_moddown(cipher.cv[1], cipher.cur_limbs, cryptoContext)
    return Cipher([res_bx, res_ax], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)



@check_meta_equal
@call_counter
def eval_add_ext(cipher0, cipher1, cryptoContext):
    moduli = cryptoContext.BsContext.QplusP_map[cipher0.cur_limbs]
    cv = [
        F.cv_add(cv0, cv1, moduli, cipher0.cur_limbs + cryptoContext.K)
        for cv0, cv1 in zip(cipher0.cv, cipher1.cv)
    ]
    return Cipher(cv, cipher0.cur_limbs, cipher0.scaling_factor, cipher0.noise_deg, cipher0.slots)

@call_counter
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

def key_switch_ext(cipher, add_first, cryptoContext):
    assert add_first == True
    cv0 = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)
    cv1 = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)

    cv0 = torch.cat((cv0, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)), dim=0)
    cv1 = torch.cat((cv1, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)), dim=0)

    return Cipher([cv0, cv1], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)


def eval_rotation_ext(cipher, digits, index, add_first, cryptoContext):

    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.find_auto_index(index)
    
    # Inner Product
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.reshape(-1), curr_limbs=cipher.cur_limbs, context=cryptoContext, swk_bx=swk[0], swk_ax=swk[1])
    sumbxmult, sumaxmult = sum_mult[0], sum_mult[1]

    if (add_first):
        cMult = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                cryptoContext.q_mu_cuda, cipher.cur_limbs)
        sumbxmult = F.cv_add(sumbxmult, cMult, cryptoContext.moduliQ_cuda, cipher.cur_limbs, inplace=True)

    cv0 = F.cv_automorphism_transform(sumbxmult, cipher.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
    cv1 = F.cv_automorphism_transform(sumaxmult, cipher.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
    return Cipher([cv0, cv1], cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)




