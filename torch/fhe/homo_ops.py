from enum import Enum
from .ciphertext import Cipher
from .ciphertext import Plaintext
from . import functional as F
from . import hybrid_keyswitch
import math
import torch

BASE_NUM_LEVELS_TO_DROP = 1


def drop_last_elements(ct, num_levels, inplace=False):
    assert num_levels <= ct.cur_limbs and num_levels >= 0
    if not inplace:
        ct = ct.deep_copy()
    ct.cur_limbs -= num_levels
    return ct


def adjust_to(cipher, target_limbs, target_noise_deg, cryptoContext):
    assert cipher.cur_limbs >= target_limbs
    if cipher.cur_limbs == target_limbs:
        if cipher.noise_deg < target_noise_deg:
            return _eval_mult_core(cipher, 1.0, cryptoContext)
        else:
            return cipher
    else:
        if cipher.noise_deg == 2 and target_noise_deg == 2:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            cipher = homo_rescale_internal(
                cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )
            if cipher.cur_limbs > target_limbs:
                cipher = drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs,
                    inplace=False,
                )
        elif cipher.noise_deg == 1 and target_noise_deg == 1:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            if cipher.cur_limbs > target_limbs + 1:
                cipher = drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs - 1,
                    inplace=True,
                )
            cipher = homo_rescale_internal(
                cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )
        elif cipher.noise_deg == 2 and target_noise_deg == 1:
            if cipher.cur_limbs == target_limbs + 1:
                homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            else:
                cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
                cipher = homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext
                )
                if cipher.cur_limbs > target_limbs + 1:
                    cipher = drop_last_elements(
                        cipher,
                        cipher.cur_limbs - target_limbs - 1,
                        inplace=True,
                    )
                cipher = homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext
                )
        elif cipher.noise_deg == 1 and target_noise_deg == 2:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            cipher = drop_last_elements(
                cipher, cipher.cur_limbs - target_limbs, inplace=True
            )
        else:
            print("noise_deg", cipher.noise_deg, target_noise_deg)
            raise ValueError
    return cipher


def adjust_levels_and_depth(ct1, ct2, cryptoContext):
    if ct1.cur_limbs > ct2.cur_limbs:
        target_limbs = ct2.cur_limbs
        target_noise_deg = ct2.noise_deg
    elif ct1.cur_limbs < ct2.cur_limbs:
        target_limbs = ct1.cur_limbs
        target_noise_deg = ct1.noise_deg
    else:
        target_limbs = ct1.cur_limbs
        target_noise_deg = max(ct1.noise_deg, ct2.noise_deg)

    return adjust_to(ct1, target_limbs, target_noise_deg, cryptoContext), adjust_to(
        ct2, target_limbs, target_noise_deg, cryptoContext
    )


def _adjust_for_add_or_sub(in0, in1, cryptoContext):
    return adjust_levels_and_depth(in0, in1, cryptoContext)


def _adjust_for_mult(ct1: Cipher, ct2: Cipher, cryptoContext):
    rct1, rct2 = adjust_levels_and_depth(ct1, ct2, cryptoContext)
    if rct1.noise_deg == 2:
        rct1 = homo_rescale_internal(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    if rct2.noise_deg == 2:
        rct2 = homo_rescale_internal(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return rct1, rct2


class LargeScalingFactorConstants(Enum):
    MAX_BITS_IN_WORD = 61
    MAX_LOG_STEP = 60


def crt_mult(xs, ys, mods):
    return [(int(x) * int(y)) % int(mod) for x, y, mod in zip(xs, ys, mods)]


def _get_element_for_eval_add_or_sub(constant, cur_limbs, noise_deg, cryptoContext):

    sc_factor = cryptoContext.approxSF
    log_approx = 0
    res = math.fabs(constant * sc_factor)
    if res > 0:
        log_sf = int(math.ceil(math.log2(res)))
        log_valid = min(log_sf, LargeScalingFactorConstants.MAX_BITS_IN_WORD.value)
        log_approx = log_sf - log_valid

    approx_factor = float(pow(2, log_approx))
    sc_constant = int(constant * sc_factor / approx_factor + 0.5)

    crt_constant = cur_limbs * [sc_constant]

    int_sc_factor = int(sc_factor + 0.5)
    crt_sc_factor = cur_limbs * [int_sc_factor]

    for i in range(1, noise_deg):
        crt_constant = crt_mult(
            crt_constant, crt_sc_factor, cryptoContext.moduliQ_scalar
        )

    return crt_constant


def _get_element_for_eval_mult(constant, cur_limbs, cryptoContext):
    sc_factor = cryptoContext.approxSF
    MAX_BITS_IN_WORD_LOCAL = 125
    log_approx = 0
    res = math.fabs(constant * sc_factor)
    if res > 0:
        log_sf = int(math.ceil(math.log2(res)))
        log_valid = (
            log_sf if log_sf <= MAX_BITS_IN_WORD_LOCAL else MAX_BITS_IN_WORD_LOCAL
        )
        log_approx = log_sf - log_valid

    approx_factor = float(pow(2, log_approx))

    large = int((constant / approx_factor * sc_factor) + 0.5)
    large_abs = abs(large)
    bound = 1 << 63

    factors = [0] * cur_limbs
    if large_abs >= bound:
        for i in range(cur_limbs):
            reduced = large % cryptoContext.moduliQ_scalar[i]
            factors[i] = (
                reduced + cryptoContext.moduliQ_scalar[i] if reduced < 0 else reduced
            )
    else:
        sc_constant = int(large)
        for i in range(cur_limbs):
            reduced = sc_constant % int(cryptoContext.moduliQ_scalar[i])
            factors[i] = (
                reduced + cryptoContext.moduliQ_scalar[i] if reduced < 0 else reduced
            )

    return factors


def _eval_mult_core(in0, cnst, cryptoContext):
    factors = _get_element_for_eval_mult(cnst, in0.cur_limbs, cryptoContext)
    return _cipher_mul_scalar_double(in0, factors, cryptoContext)


def _cipher_add(in0, in1, cryptoContext):
    cv = [
        F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


def _cipher_add_ext(in0, in1, cryptoContext):
    cv = [
        F.cv_add(
            cv0,
            cv1,
            cryptoContext.BsContext.QplusP_map[in0.cur_limbs],
            in0.cur_limbs + cryptoContext.K,
        )
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


def _cipher_sub(in0, in1, cryptoContext):
    cv = [
        F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


def _cipher_mul(in0, in1, cryptoContext):
    bx = F.cv_mul(
        in0.cv[0],
        in1.cv[0],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        in0.cur_limbs,
    )
    ax = F.cv_add(
        F.cv_mul(
            in0.cv[0],
            in1.cv[1],
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        ),
        F.cv_mul(
            in0.cv[1],
            in1.cv[0],
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        ),
        cryptoContext.moduliQ,
        in0.cur_limbs,
    )
    axax = F.cv_mul(
        in0.cv[1],
        in1.cv[1],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        in0.cur_limbs,
    )
    return in0.cipher_like(
        [bx, ax, axax],
        noise_deg=in0.noise_deg + in1.noise_deg,
    )


def _cipher_square(in0, cryptoContext):
    bx = F.cv_mul(
        in0.cv[0],
        in0.cv[0],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        in0.cur_limbs,
    )
    ax = F.cv_mul(
        in0.cv[0],
        in0.cv[1],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        in0.cur_limbs,
    )
    ax = F.cv_add(ax, ax, cryptoContext.moduliQ, in0.cur_limbs)
    axax = F.cv_mul(
        in0.cv[1],
        in0.cv[1],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        in0.cur_limbs,
    )
    return in0.cipher_like(
        [bx, ax, axax],
        noise_deg=in0.noise_deg + in0.noise_deg,
    )


def _cipher_add_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(
        scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs
    )
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


def _cipher_sub_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(
        scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs
    )
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


def _cipher_mul_scalar_double(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(
        scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs
    )
    cv = [
        F.cv_mul_scalar(
            cv0,
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        )
        for cv0 in in0.cv
    ]
    return in0.cipher_like(cv, noise_deg=in0.noise_deg + 1)


def _cipher_mul_scalar_int(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(
        scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs
    )
    cv = [
        F.cv_mul_scalar(
            cv0,
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        )
        for cv0 in in0.cv
    ]
    return in0.cipher_like(cv, noise_deg=in0.noise_deg)


def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return in0.cipher_like(cv, noise_deg=in0.noise_deg)


def _cipher_automorphism(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    limbs = in0.cur_limbs if in0.is_ext == False else in0.cur_limbs + cryptoContext.K
    cv = [
        F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext)
        for cv in in0.cv
    ]
    return in0.cipher_like(cv)


def homo_add(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_add_ext(in0, in1, cryptoContext)
    else:
        return _cipher_add(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)


def homo_rescale_internal(ct, levels, cryptoContext):
    assert levels == 1 or levels == 0 and "Only support these two cases"
    if levels == 0:
        return ct.deep_copy()

    def rescale_n_times(cv, levels):
        for l in range(levels):
            cv = F.cv_drop_last_element_and_scale(cv, ct.cur_limbs, l, cryptoContext)
        return cv

    res_cv = [rescale_n_times(_cv, levels) for _cv in ct.cv]

    return ct.cipher_like(
        res_cv,
        cur_limbs=ct.cur_limbs - levels,
        noise_deg=ct.noise_deg - levels,
    )


def homo_mul(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_mult(in0, in1, cryptoContext)
    res = _cipher_mul(in0, in1, cryptoContext)
    tmp = res.cipher_like(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx,
            cryptoContext.swk_ax,
            cryptoContext,
        )
    )
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)


def homo_square(in0, cryptoContext):
    if in0.noise_deg != 1:
        in0 = homo_rescale_internal(in0, 1, cryptoContext)
    res = _cipher_square(in0, cryptoContext)
    tmp = res.cipher_like(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx,
            cryptoContext.swk_ax,
            cryptoContext,
        )
    )
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)


def homo_add_scalar_double(in0, cnst, cryptoContext, precomp=None):
    tmpr = _get_element_for_eval_add_or_sub(
        math.fabs(cnst), in0.cur_limbs, in0.noise_deg, cryptoContext
    )
    if cnst < 0:
        return _cipher_sub_scalar(in0, tmpr, cryptoContext)
    else:
        return _cipher_add_scalar(in0, tmpr, cryptoContext)


def homo_add_scalar_int(in0, scalar, cryptoContext):
    return _cipher_add_scalar(in0, scalar, cryptoContext)


def homo_mul_scalar_int(in0, scalar, cryptoContext):
    res = _cipher_mul_scalar_int(in0, abs(scalar), cryptoContext)
    if scalar < 0:
        res = _cipher_neg(res, cryptoContext)
    return res


def homo_mul_scalar_double(in0, cnst, cryptoContext):
    if in0.noise_deg == 2:
        in0 = homo_rescale_internal(in0, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return _eval_mult_core(in0, cnst, cryptoContext)


def homo_rotate(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.left_rot_key_map[norm_index]
    res = in0.cipher_like(
        F.cv_keyswitch(in0.cv[1], in0.cur_limbs, swk[0], swk[1], cryptoContext)
    )

    res.cv[0] = F.cv_add(in0.cv[0], res.cv[0], cryptoContext.moduliQ, in0.cur_limbs)
    res = _cipher_automorphism(res, index, cryptoContext)

    return res


def eval_fast_rotate(digits, cipher, index, need_KS_add, need_moddown, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    result = hybrid_keyswitch.mult_rot_key_and_sum_ext(digits, index, cryptoContext)

    if need_KS_add:
        if need_moddown:
            result = hybrid_keyswitch.moddown_from_ext(result, cryptoContext)
            cipher_cv0 = cipher.cv[0]
        else:
            cipher_cv0 = F.cv_mul_scalar(
                cipher.cv[0],
                cryptoContext.PModq,
                cryptoContext.moduliQ,
                cryptoContext.q_mu,
                cipher.cur_limbs,
            )

        result.cv[0] = F.cv_add(
            result.cv[0],
            cipher_cv0,
            cryptoContext.moduliQ,
            cipher.cur_limbs,
            inplace=True,
        )

    result = _cipher_automorphism(result, index, cryptoContext)

    return result


def homo_conjugate(in0, cryptoContext):
    return homo_rotate(in0, 2 * cryptoContext.N - 1, cryptoContext)


def homo_add_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    ctmorphed = plaintext.cipher_like(plaintext.cv)
    res0, res1 = _adjust_for_add_or_sub(cipher, ctmorphed, cryptoContext)
    res0.cv = [
        F.cv_add(res0.cv[0], res1.cv[0], cryptoContext.moduliQ, res0.cur_limbs),
        res0.cv[1],
    ]
    return res0


def homo_mul_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    assert len(cipher.cv) == 2

    if cipher.is_ext:
        if (
            cipher.cur_limbs != plaintext.cur_limbs
            or cipher.noise_deg != plaintext.noise_deg
            or cipher.is_ext != plaintext.is_ext
        ):
            raise ValueError(
                f"Unequal values! Cipher and plaintext have mismatched properties:\n"
                f"  cipher.cur_limbs = {cipher.cur_limbs}, plaintext.cur_limbs = {plaintext.cur_limbs}\n"
                f"  cipher.noise_deg = {cipher.noise_deg}, plaintext.noise_deg = {plaintext.noise_deg}\n"
                f"  cipher.is_ext = {cipher.is_ext}, plaintext.is_ext = {plaintext.is_ext}"
            )
        moduli = cryptoContext.BsContext.QplusP_map[cipher.cur_limbs]
        mu = cryptoContext.BsContext.QmuplusPmu_map[cipher.cur_limbs]
        cv0 = F.cv_mul(
            cipher.cv[0], plaintext.cv, moduli, mu, cipher.cur_limbs + cryptoContext.K
        )
        cv1 = F.cv_mul(
            cipher.cv[1], plaintext.cv, moduli, mu, cipher.cur_limbs + cryptoContext.K
        )
        return cipher.cipher_like(
            [cv0, cv1],
            noise_deg=cipher.noise_deg + plaintext.noise_deg,
        )
    else:
        ctmorphed = plaintext.cipher_like(plaintext.cv)
        res0, res1 = _adjust_for_mult(cipher, ctmorphed, cryptoContext)

        moduli = cryptoContext.moduliQ
        mu = cryptoContext.q_mu
        cv0 = F.cv_mul(res0.cv[0], res1.cv[0], moduli, mu, res0.cur_limbs)
        cv1 = F.cv_mul(res0.cv[1], res1.cv[0], moduli, mu, res0.cur_limbs)

        return res0.cipher_like(
            [cv0, cv1],
            noise_deg=res0.noise_deg + res1.noise_deg,
        )


def extract_cv(cipher: Cipher, index, append_zeros=False):
    assert index == 0 or index == 1, "index must be 0 or 1"
    if append_zeros:
        if index == 0:
            return cipher.cipher_like([cipher.cv[0], torch.zeros_like(cipher.cv[0])])
        else:
            return cipher.cipher_like([torch.zeros_like(cipher.cv[1]), cipher.cv[1]])
    else:
        return cipher.cipher_like([cipher.cv[index]])
