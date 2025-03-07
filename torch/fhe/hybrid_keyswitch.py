from . import functional as F
import torch


def key_switch_P_ext(cipher, cryptoContext):
    assert cipher.is_ext == False
    cv = [
        torch.cat(
            (
                F.cv_mul_scalar(
                    cv,
                    cryptoContext.PModq,
                    cryptoContext.moduliQ,
                    cryptoContext.q_mu,
                    cipher.cur_limbs,
                ),
                torch.zeros(
                    (cryptoContext.K << cryptoContext.logN),
                    dtype=torch.uint64,
                    device="cuda",
                ).reshape(-1, cryptoContext.N),
            ),
            dim=0,
        )
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=True)


def modup_to_ext(cipher, cryptoContext):
    assert cipher.is_ext == False
    assert len(cipher.cv) == 1
    cv = [F.cv_modup(cipher.cv[0], cipher.cur_limbs, cryptoContext)]
    return cipher.cipher_like(cv, is_ext=True)


def mult_rot_key_and_sum_ext(digits, index, cryptoContext):
    assert digits.is_ext == True
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.left_rot_key_map[norm_index]
    sum_mult = F.cv_innerproduct(
        digits.cv[0].reshape(-1),
        curr_limbs=digits.cur_limbs,
        context=cryptoContext,
        swk_bx=swk[0],
        swk_ax=swk[1],
    )
    return digits.cipher_like(sum_mult, is_ext=True)


def moddown_from_ext(cipher, cryptoContext):
    assert cipher.is_ext == True
    cv = [F.cv_moddown(cv, cipher.cur_limbs, cryptoContext) for cv in cipher.cv]
    return cipher.cipher_like(cv, is_ext=False)
