from .ciphertext import Cipher
from . import functional as F
import torch


def key_switch_ext(cipher, cryptoContext):
    assert cipher.is_ext == False
    cv_len = len(cipher.cv)
    cv = [None] * cv_len
    for i in range(cv_len):
        cv[i] = F.cv_mul_scalar(cipher.cv[i], cryptoContext.PModq, cryptoContext.moduliQ,
                                cryptoContext.q_mu, cipher.cur_limbs)
        cv[i] = torch.cat((cv[i], torch.zeros((cryptoContext.K << cryptoContext.logN), dtype=torch.uint64,
                                              device="cuda").reshape(-1, cryptoContext.N)), dim=0)
    return cipher.cipher_like(cv, is_ext=True)

def modup_to_ext(cipher, cryptoContext):
    assert cipher.is_ext == False
    cv = [
        F.cv_modup(cv, cipher.cur_limbs, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext = True)


def mult_key_and_sum_ext(digits, cipher, index, cryptoContext):
    assert digits.is_ext == True
    assert cipher.is_ext == False

    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.find_auto_index(index)

    # Inner Product
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), curr_limbs=digits.cur_limbs, context=cryptoContext,
                                 swk_bx=swk[0], swk_ax=swk[1])

    return cipher.cipher_like(sum_mult.clone(), is_ext = True) #fixme: delete clone()?


def moddown_from_ext(cipher, cryptoContext):
    assert cipher.is_ext == True
    cv = [
        F.cv_moddown(cv, cipher.cur_limbs, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=False)


def eval_fast_rotation(cipher, index, digits, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    cur_limbs = cipher.cur_limbs

    # EvalFastKeySwitchCore = InnerProduct + ModDown
    auto_index = cryptoContext.find_auto_index(index)
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    sum_mult = F.cv_innerproduct(digits.cv[0].reshape(-1), curr_limbs=digits.cur_limbs, context=cryptoContext,
                                 swk_bx=swk[0], swk_ax=swk[1])
    sumMult = Cipher(sum_mult, cipher.cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots, is_ext=True)
    result = moddown_from_ext(sumMult, cryptoContext)
    # post add after ks
    result.cv[0] = F.cv_add(cipher.cv[0], result.cv[0], cryptoContext.moduliQ, cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    result.cv[0] = F.cv_automorphism_transform(result.cv[0], cur_limbs, auto_index, cryptoContext)
    result.cv[1] = F.cv_automorphism_transform(result.cv[1], cur_limbs, auto_index, cryptoContext)

    return result