import numpy as np
from .ciphertext import Cipher
from . import functional as F
from .utils import check_meta_equal
import torch

def mult_rot_key_and_sum_ext(digits, index, cryptoContext):
    assert digits.is_ext == True
    auto_index = cryptoContext.find_auto_index(index)

    # Inner Product
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), digits.cur_limbs, swk[0], swk[1], cryptoContext)
    sumbxmult, sumaxmult = sum_mult[0], sum_mult[1]
    return digits.cipher_like(sum_mult, is_ext=True)
#todo: it is ct*pt in extent form, refactor?
def eval_mult_ext(cipher, pt, cryptoContext):
    moduli = cryptoContext.BsContext.QplusP_map[cipher.cur_limbs]
    mu = cryptoContext.BsContext.QmuplusPmu_map[cipher.cur_limbs]
    cv0 = F.cv_mul(cipher.cv[0], pt.mv, moduli, mu, cipher.cur_limbs + cryptoContext.K)
    cv1 = F.cv_mul(cipher.cv[1], pt.mv, moduli, mu, cipher.cur_limbs + cryptoContext.K)
    return cipher.cipher_like([cv0, cv1], scaling_factor=cipher.scaling_factor*pt.scaling_factor, noise_deg=cipher.noise_deg+pt.noise_deg)

def key_switch_ext(cipher, cryptoContext):
    assert cipher.is_ext == False

    cv0 = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)
    cv1 = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)

    cv0 = torch.cat((cv0, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device=cv0.device).reshape(-1, cryptoContext.N)), dim=0)
    cv1 = torch.cat((cv1, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device=cv1.device).reshape(-1, cryptoContext.N)), dim=0)

    return cipher.cipher_like([cv0, cv1], is_ext=True)

def modup_to_ext(cipher, cryptoContext):
    assert cipher.is_ext == False
    cv = [
        F.cv_modup(cv, cipher.cur_limbs, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext = True)

def moddown_from_ext(cipher, cryptoContext):
    assert cipher.is_ext == True
    cv = [
        F.cv_moddown(cv, cipher.cur_limbs, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=False)

def eval_automorphism(cipher, index, cryptoContext):
    assert cipher.is_ext == False
    auto_index = cryptoContext.find_auto_index(index)
    cv = [
        F.cv_automorphism_transform(cv, cipher.cur_limbs, auto_index, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv)

def fused_rotation_add_ext(digits, cipher, index, cryptoContext):

    assert digits.is_ext == True
    assert cipher.is_ext == False

    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.find_auto_index(index)
    
    # Inner Product
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), digits.cur_limbs, swk[0], swk[1], cryptoContext)
    sumbxmult, sumaxmult = sum_mult[0], sum_mult[1]

    cMult = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                            cryptoContext.q_mu_cuda, cipher.cur_limbs)
    sumbxmult = F.cv_add(sumbxmult, cMult, cryptoContext.moduliQ_cuda, cipher.cur_limbs, inplace=True)

    cv0 = F.cv_automorphism_transform(sumbxmult, digits.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
    cv1 = F.cv_automorphism_transform(sumaxmult, digits.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
    return digits.cipher_like([cv0, cv1], is_ext=True)


def eval_fast_rotation(ciphertext, index, digits, cryptoContext):
    if index == 0:
        return ciphertext.deep_copy()

    cur_limbs = ciphertext.cur_limbs

    # EvalFastKeySwitchCore = InnerProduct + ModDown
    auto_index = cryptoContext.find_auto_index(index)
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), digits.cur_limbs, swk[0], swk[1], cryptoContext)
    sumMult = Cipher(sum_mult, ciphertext.cur_limbs, ciphertext.scaling_factor, ciphertext.noise_deg, ciphertext.slots, is_ext=True)
    result = moddown_from_ext(sumMult, cryptoContext)
    # post add after ks
    result.cv[0] = F.cv_add(ciphertext.cv[0], result.cv[0], cryptoContext.moduliQ_cuda, cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    result.cv[0] = F.cv_automorphism_transform(result.cv[0], cur_limbs, auto_index, cryptoContext)
    result.cv[1] = F.cv_automorphism_transform(result.cv[1], cur_limbs, auto_index, cryptoContext)

    return result