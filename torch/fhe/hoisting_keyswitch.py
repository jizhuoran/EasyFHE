from .ciphertext import Cipher
from . import functional as F
import torch


def key_switch_ext(cipher, cryptoContext):
    assert cipher.is_ext == False

    cv0 = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)
    cv1 = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher.cur_limbs)

    cv0 = torch.cat((cv0, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)), dim=0)
    cv1 = torch.cat((cv1, torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)), dim=0)

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


def eval_fast_rotation(ciphertext, index, digits, cryptoContext):
    if index == 0:
        return ciphertext.deep_copy()

    cur_limbs = ciphertext.cur_limbs

    # EvalFastKeySwitchCore = InnerProduct + ModDown
    auto_index = cryptoContext.find_auto_index(index)
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), curr_limbs=digits.cur_limbs, context=cryptoContext,
                                 swk_bx=swk[0], swk_ax=swk[1])
    sumMult = Cipher(sum_mult, ciphertext.cur_limbs, ciphertext.scaling_factor, ciphertext.noise_deg, ciphertext.slots, is_ext=True)
    result = moddown_from_ext(sumMult, cryptoContext)
    # post add after ks
    result.cv[0] = F.cv_add(ciphertext.cv[0], result.cv[0], cryptoContext.moduliQ_cuda, cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    result.cv[0] = F.cv_automorphism_transform(result.cv[0], cur_limbs, auto_index, cryptoContext)
    result.cv[1] = F.cv_automorphism_transform(result.cv[1], cur_limbs, auto_index, cryptoContext)

    return result