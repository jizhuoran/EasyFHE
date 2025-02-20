from .ciphertext import Cipher
from . import functional as F
import torch


def key_switch_ext(cipher, cryptoContext):
    assert cipher.is_ext == False
    cv = [
        torch.cat((
            F.cv_mul_scalar(cv, cryptoContext.PModq, cryptoContext.moduliQ, cryptoContext.q_mu, cipher.cur_limbs),
            torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
        ), dim=0)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=True)

def modup_to_ext(cipher, cryptoContext):
    assert cipher.is_ext == False
    assert len(cipher.cv) == 1
    cv = [F.cv_modup(cipher.cv[0], cipher.cur_limbs, cryptoContext)]
    return cipher.cipher_like(cv, is_ext = True)


def mult_rot_key_and_sum_ext(digits, cipher, index, cryptoContext):
    assert digits.is_ext == True
    assert cipher.is_ext == False
    auto_index = cryptoContext.find_auto_index(index)
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), curr_limbs=digits.cur_limbs, context=cryptoContext,
                                 swk_bx=swk[0], swk_ax=swk[1])
    return cipher.cipher_like(sum_mult, is_ext = True)


def moddown_from_ext(cipher, cryptoContext):
    assert cipher.is_ext == True
    cv = [
        F.cv_moddown(cv, cipher.cur_limbs, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=False)


def eval_fast_rotation(digits, cipher, index, need_moddown, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    cur_limbs = cipher.cur_limbs

    # EvalFastKeySwitchCore = InnerProduct + ModDown
    auto_index = cryptoContext.find_auto_index(index)

    sumMult = mult_rot_key_and_sum_ext(digits, cipher, index, cryptoContext)

    if need_moddown:
        result = moddown_from_ext(sumMult, cryptoContext)
        # post add after ks
        result.cv[0] = F.cv_add(result.cv[0], cipher.cv[0], cryptoContext.moduliQ, cur_limbs)

        # Apply the AutomorphismTransform to ax and bx
        result.cv[0] = F.cv_automorphism_transform(result.cv[0], cur_limbs, auto_index, cryptoContext)
        result.cv[1] = F.cv_automorphism_transform(result.cv[1], cur_limbs, auto_index, cryptoContext)

        return result
    else:
        cipher_pmodup_cv0 = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq, cryptoContext.moduliQ,
                                            cryptoContext.q_mu, cipher.cur_limbs)
        # operate first sumMult.cv[0][:curr_limbs], sumMult.cv[0][curr_limbs+1:] remain unchanged
        sumMult.cv[0] = F.cv_add(sumMult.cv[0], cipher_pmodup_cv0, cryptoContext.moduliQ, cipher.cur_limbs, inplace=True)

        sumMult.cv[0] = F.cv_automorphism_transform(sumMult.cv[0], digits.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
        sumMult.cv[1] = F.cv_automorphism_transform(sumMult.cv[1], digits.cur_limbs + cryptoContext.K, auto_index, cryptoContext)
        return sumMult
