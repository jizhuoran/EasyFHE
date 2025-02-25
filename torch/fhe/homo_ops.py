from .ciphertext import Cipher
from .ciphertext import Plaintext
from . import functional as F
from . import hybrid_keyswitch
import math
import torch
from .utils import check_meta_equal, check_cipher_len, call_counter, profile_python_function
import warnings


BASE_NUM_LEVELS_TO_DROP = 1  # todo: to be removed?


# note: AdjustLevelsInPlace in rns-leveledshe.cpp
# mainly for "FIXEDMANUAL" case
def _adjust_levels(ct1, ct2, cryptoContext):
    ct1, ct2 = ct1.shallow_copy(), ct2.shallow_copy()
    if ct1.cur_limbs > ct2.cur_limbs:
        ct1.drop_last_elements(ct1.cur_limbs - ct2.cur_limbs)
    elif ct1.cur_limbs < ct2.cur_limbs:
        ct2.drop_last_elements(ct2.cur_limbs - ct1.cur_limbs)
    return ct1, ct2


# note: AdjustLevelsAndDepthToOneInPlace in rns-leveledshe.cpp
def adjust_levels_and_depth(ct1, ct2, cryptoContext):
    if ct1.cur_limbs < ct2.cur_limbs:
        rct1, rct2, swapped = ct2.shallow_copy(), ct1.shallow_copy(), True
    else:
        rct1, rct2, swapped = ct1.shallow_copy(), ct2.shallow_copy(), False

    if not rct1.cur_limbs == rct2.cur_limbs:
        scf1 = rct1.scaling_factor
        scf2 = cryptoContext.GetScalingFactorRealBig(
            rct2.cur_limbs + 2 - rct2.noise_deg
        )
        scf = cryptoContext.GetScalingFactorReal(rct1.cur_limbs)
        q1 = (
            cryptoContext.GetModReduceFactor(rct1.cur_limbs - 1)
            if rct1.noise_deg == 2
            else 1
        )
        scaling_factor = scf2 / scf1 * q1 / scf
        if rct1.noise_deg == 2 and not (
            rct2.noise_deg == 1 and rct1.cur_limbs - rct2.cur_limbs == 1
        ):
            rct1 = _eval_mult_core(rct1, scaling_factor, cryptoContext)
            rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        elif rct1.noise_deg == 1:
            rct1 = _eval_mult_core(rct1, scaling_factor, cryptoContext)
        else:
            raise ValueError
        rct1.try_drop_last_elements(
            rct1.cur_limbs - rct2.cur_limbs + rct2.noise_deg - 2
        )
        if rct2.noise_deg == 1:
            rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        rct1.scaling_factor = rct2.scaling_factor

        if swapped:
            rct1, rct2 = rct2.shallow_copy(), rct1.shallow_copy()
    else:
        if rct1.noise_deg < rct2.noise_deg:
            rct1 = _eval_mult_core(rct1, 1.0, cryptoContext)
        elif rct2.noise_deg < rct1.noise_deg:
            rct2 = _eval_mult_core(rct2, 1.0, cryptoContext)
    return rct1, rct2


# AdjustForAddOrSubInPlace in rns-leveledshe.cpp
def _adjust_for_add_or_sub(in0, in1, cryptoContext):
    rescaleTech = cryptoContext.rescaleTech
    if rescaleTech == "FIXEDMANUAL":
        # function `_adjust_levels` doesnt change memory, therefore its fine with a ciphertext morphed form plaintext
        in0, in1 = _adjust_levels(in0, in1, cryptoContext)
        if len(in0.cv) == len(in1.cv):
            return in0, in1

        scFactor = cryptoContext.GetScalingFactorReal(
            cryptoContext.L
        )  # openfhe default value is 0, here transfer to the max value of #limb
        if scFactor == 0.0:
            raise ValueError("Unsupported scaling factor")

        if len(in0.cv) == 1:
            ptxt = in0.cv[0]
            ptxtDepth = in0.noise_deg
            ctxtDepth = in1.noise_deg
            sizeQl = in1.cur_limbs
            moduli = cryptoContext.moduliQ_scalar[:sizeQl]
            ptxtIndex = 0
        elif len(in1.cv) == 1:
            ptxt = in1.cv[0]
            ptxtDepth = in1.noise_deg
            ctxtDepth = in0.noise_deg
            sizeQl = in0.cur_limbs
            moduli = cryptoContext.moduliQ_scalar[:sizeQl]
            ptxtIndex = 1

        if len(in0.cv) == 1 or len(in1.cv) == 1:  # todo: this branch is not tested
            # Bring to same depth if not already same
            if ptxtDepth < ctxtDepth:
                diffDepth = ctxtDepth - ptxtDepth
                intSF = int(scFactor + 0.5)  # todo: to check if equivalent to openfhe
                crtSF = torch.tensor(sizeQl * [intSF], dtype=torch.uint64)
                crtPowSF = torch.clone(crtSF)
                for i in range(diffDepth):
                    crtPowSF = crt_mult(crtPowSF, crtSF, moduli)
                ptxt = F.cv_mul_scalar(
                    ptxt,
                    crtPowSF.cuda(),
                    cryptoContext.moduliQ,
                    cryptoContext.q_mu,
                    len(moduli),
                )  # fixme: crtPowSF should be a tensor for F.cv_mul_scalar? refactor crt_mult?

                if ptxtIndex == 0:
                    in0.cv[0] = ptxt  # todo: check if correctly assigned
                    in0.noise_deg = ctxtDepth
                else:
                    in1.cv[0] = ptxt  # todo: check if correctly assigned
                    in1.noise_deg = ctxtDepth
            elif ptxtDepth > ctxtDepth:
                raise ValueError(
                    "plaintext cannot be encoded at a larger depth than that of the ciphertext."
                )

    else:
        in0, in1 = adjust_levels_and_depth(in0, in1, cryptoContext)

    return in0, in1

def _adjust_for_mult(ct1: Cipher, ct2: Cipher, cryptoContext):
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        rct1, rct2 = _adjust_levels(ct1, ct2, cryptoContext)
    else:
        # inline `AdjustLevelsAndDepthToOneInPlace` in ckksrns-leveledshe.cpp as following
        rct1, rct2 = adjust_levels_and_depth(ct1, ct2, cryptoContext)
        if rct1.noise_deg == 2:
            rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return rct1, rct2


from enum import Enum


class LargeScalingFactorConstants(Enum):
    MAX_BITS_IN_WORD = 61
    MAX_LOG_STEP = 60


# CRTMult in ckkspackedencoding.cpp
def crt_mult(xs, ys, mods):
    return [(int(x) * int(y)) % int(mod) for x, y, mod in zip(xs, ys, mods)]


# todo: implement void EvalSubInPlace(Ciphertext<Element>& ciphertext, double constant) in cryptocontext.h?
def _get_element_for_eval_add_or_sub(constant, cur_limbs, noise_deg, cryptoContext):

    if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT" and cur_limbs == cryptoContext.L:
        sc_factor = cryptoContext.GetScalingFactorRealBig(cur_limbs)
    else:
        sc_factor = cryptoContext.GetScalingFactorReal(cur_limbs)

    # Compute approxFactor to avoid overflow issues
    log_approx = 0
    res = math.fabs(constant * sc_factor)
    if res > 0:
        log_sf = int(math.ceil(math.log2(res)))
        log_valid = min(log_sf, LargeScalingFactorConstants.MAX_BITS_IN_WORD.value)
        log_approx = log_sf - log_valid

    approx_factor = float(pow(2, log_approx))
    sc_constant = int(constant * sc_factor / approx_factor + 0.5)

    crt_constant = cur_limbs * [sc_constant]

    # Scale back up by approxFactor within the CRT multiplications.
    if log_approx > 0:
        log_step = min(log_approx, LargeScalingFactorConstants.MAX_LOG_STEP.value)
        int_step = 2**log_step
        crt_approx = cur_limbs * [int_step]
        log_approx -= log_step

        while log_approx > 0:
            log_step = min(log_approx, LargeScalingFactorConstants.MAX_LOG_STEP.value)
            int_step = 2**log_step
            crt_sf = cur_limbs * [int_step]
            crt_approx = crt_mult(crt_approx, crt_sf, cryptoContext.moduliQ_scalar)
            log_approx -= log_step

        crt_constant = crt_mult(crt_constant, crt_approx, cryptoContext.moduliQ_scalar)

    # Handle FLEXIBLEAUTOEXT mode at level 0, we don't use the depth to calculate the scaling factor,
    # so we return the value before taking the depth into account.
    if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT" and cur_limbs == cryptoContext.L:
        return crt_constant

    # Final scaling factor adjustments
    int_sc_factor = int(sc_factor + 0.5)
    crt_sc_factor = cur_limbs * [int_sc_factor]

    for i in range(1, noise_deg):
        crt_constant = crt_mult(crt_constant, crt_sc_factor, cryptoContext.moduliQ_scalar)

    return crt_constant


# note: GetElementForEvalMult in ckksrns-leveledshe.cpp
def _get_element_for_eval_mult(constant, cur_limbs, cryptoContext):
    sc_factor = cryptoContext.GetScalingFactorReal(cur_limbs)

    # note: Assuming DoubleInteger is equivalent to Python's int (arbitrary precision)
    MAX_BITS_IN_WORD_LOCAL = 125

    # Compute approxFactor, a value to scale down by, in case the value exceeds a 64-bit integer.
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
            factors[i] = reduced + cryptoContext.moduliQ_scalar[i] if reduced < 0 else reduced
    else:
        sc_constant = int(large)
        for i in range(cur_limbs):
            reduced = sc_constant % int(cryptoContext.moduliQ_scalar[i])
            factors[i] = reduced + cryptoContext.moduliQ_scalar[i] if reduced < 0 else reduced

    # Scale back up by approxFactor within the CRT multiplications.
    if log_approx > 0:
        log_step = (
            log_approx
            if log_approx <= LargeScalingFactorConstants.MAX_LOG_STEP.value
            else LargeScalingFactorConstants.MAX_LOG_STEP.value
        )
        int_step = 1 << log_step
        crt_approx = cur_limbs * [int_step]
        log_approx -= log_step

        while log_approx > 0:
            log_step = (
                log_approx
                if log_approx <= LargeScalingFactorConstants.MAX_LOG_STEP.value
                else LargeScalingFactorConstants.MAX_LOG_STEP.value
            )
            int_step = 1 << log_step
            crt_sf = cur_limbs * [int_step]
            crt_approx = crt_mult(crt_approx, crt_sf, cryptoContext.moduliQ_scalar)
            log_approx -= log_step
        factors = crt_mult(factors, crt_approx, cryptoContext.moduliQ_scalar)

    return factors


# note: EvalMultCoreInPlace in ckksrns-leveledshe.cpp
def _eval_mult_core(in0, cnst, cryptoContext):
    factors = _get_element_for_eval_mult(cnst, in0.cur_limbs, cryptoContext)
    return _cipher_mul_scalar_double(in0, factors, cryptoContext)

# todo: only support in `FIXEDMANUAL` mode, or `adjust_levels_and_depth` function.
# should not be used directly in other rescale modes!!! except when openfhe directly used it
# todo: write homo_level_reduce


@check_meta_equal
def _cipher_add(in0, in1, cryptoContext):
    cv = [
        F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


@check_meta_equal
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


@check_meta_equal
def _cipher_sub(in0, in1, cryptoContext):
    cv = [
        F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


@check_meta_equal
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
    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return in0.cipher_like([bx, ax, axax], scaling_factor=in0.scaling_factor * scFactor,
        noise_deg=in0.noise_deg + in1.noise_deg,
    )


@check_cipher_len
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
    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return in0.cipher_like([bx, ax, axax], scaling_factor=in0.scaling_factor * scFactor,
        noise_deg=in0.noise_deg + in0.noise_deg,
    )


@check_cipher_len
def _cipher_add_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    cv = [
        F.cv_add_scalar(
            in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs
        ),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


@check_cipher_len
def _cipher_sub_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    cv = [
        F.cv_sub_scalar(
            in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs
        ),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


# todo: used only in `homo_mul_scalar_int`, therefore the scaling factor and noise_deg remain unchanged
# todo: if used for `homo_mul_scalar_double`, the scaling factor and noise_deg should be changed
# @check_cipher_len #fixme: comment it to support call from homo_mul_pt
def _cipher_mul_scalar_double(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
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
    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor * scFactor, noise_deg=in0.noise_deg + 1)


@check_cipher_len
def _cipher_mul_scalar_int(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
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
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)


@check_cipher_len
def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)


# @check_cipher_len
#todo: input len of in0.cv could be 1
def _cipher_automorphism(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    limbs = in0.cur_limbs if in0.is_ext == False else in0.cur_limbs + cryptoContext.K
    cv = [
        F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext)
        for cv in in0.cv
    ]
    return in0.cipher_like(cv)


@call_counter
def homo_add(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_add_ext(in0, in1, cryptoContext)
    else:
        return _cipher_add(in0, in1, cryptoContext)


@call_counter
def homo_sub(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)


@call_counter
def homo_rescale(ct, levels, cryptoContext):
    assert levels == 1 or levels == 0 and "Only support these two cases"
    if levels == 0:
        return ct.deep_copy()

    def rescale_n_times(cv, levels):
        for l in range(levels):
            cv = F.cv_drop_last_element_and_scale(cv, ct.cur_limbs, l, cryptoContext)
        return cv
    
    res_cv = [rescale_n_times(_cv, levels) for _cv in ct.cv]

    scFactor = ct.scaling_factor
    for l in range(levels):
        modReduceFactor = cryptoContext.GetModReduceFactor(
            ct.cur_limbs - 1 - l
        )  # corresponding to openfhe: (sizeQl -1 -i), we need to use the value of input ct
        scFactor = scFactor / modReduceFactor

    return ct.cipher_like(res_cv, cur_limbs=ct.cur_limbs - levels, scaling_factor=scFactor,
                          noise_deg=ct.noise_deg - levels)


@call_counter
def homo_mul(in0, in1, cryptoContext):
    # note: AdjustForMultInPlace in rns-leveledshe.cpp
    in0, in1 = _adjust_for_mult(in0, in1, cryptoContext)
    res = _cipher_mul(in0, in1, cryptoContext)
    tmp = res.cipher_like(F.cv_keyswitch(
        res.cv[2],
        res.cur_limbs,
        cryptoContext.swk_bx,
        cryptoContext.swk_ax,
        cryptoContext,
    ))
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)


@call_counter
def homo_square(in0, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg != 1:
        in0 = homo_rescale(in0, 1, cryptoContext)
    res = _cipher_square(in0, cryptoContext)
    tmp = res.cipher_like(F.cv_keyswitch(
        res.cv[2],
        res.cur_limbs,
        cryptoContext.swk_bx,
        cryptoContext.swk_ax,
        cryptoContext,
    ))
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)


def homo_add_scalar_double(in0, cnst, cryptoContext, precomp = None):
    if precomp is not None:
        tmpr = precomp
    else:
        tmpr = _get_element_for_eval_add_or_sub(
            math.fabs(cnst), in0.cur_limbs, in0.noise_deg, cryptoContext
        )
    if cnst < 0:
        return _cipher_sub_scalar(in0, tmpr, cryptoContext)
    else:
        return _cipher_add_scalar(in0, tmpr, cryptoContext)


def homo_add_scalar_int(in0, scalar, cryptoContext):
    return _cipher_add_scalar(in0, scalar, cryptoContext)


# note: corresponds to MultByIntegerInPlace in openfhe, the datatype of scalar in openfhe is `uint64_t`
# fixme: do we accept scalar<0?
def homo_mul_scalar_int(in0, scalar, cryptoContext):
    res = _cipher_mul_scalar_int(in0, abs(scalar), cryptoContext)
    if scalar < 0:
        res = _cipher_neg(res, cryptoContext)
    return res


# note: EvalMultInPlace in ckksrns-leveledshe.cpp
@profile_python_function
@call_counter
def homo_mul_scalar_double(in0, cnst, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg == 2:
        in0 = homo_rescale(in0, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return _eval_mult_core(in0, cnst, cryptoContext)


def homo_rotate(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.left_rot_key_map[norm_index]
    res = in0.cipher_like(F.cv_keyswitch(in0.cv[1], in0.cur_limbs, swk[0], swk[1], cryptoContext))

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
            cipher_cv0 = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq, cryptoContext.moduliQ,
                                         cryptoContext.q_mu, cipher.cur_limbs) # PModUp

        # post add after ks
        # if need_moddown = False, operate sumMult.cv[0][:curr_limbs], and sumMult.cv[0][curr_limbs+1:] remain unchanged,
        # so the `inplace` can't be removed trivially
        result.cv[0] = F.cv_add(result.cv[0], cipher_cv0, cryptoContext.moduliQ, cipher.cur_limbs, inplace=True)

    result = _cipher_automorphism(result, index, cryptoContext)

    return result


def homo_conjugate(in0, cryptoContext):
    return homo_rotate(in0, 2 * cryptoContext.N - 1, cryptoContext)


def homo_add_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    # res0 = cipher.deep_copy()
    ctmorphed = plaintext.cipher_like(plaintext.mv) # MorphPlaintext in openfhe
    res0, res1 = _adjust_for_add_or_sub(cipher, ctmorphed, cryptoContext)
    res0.cv = [
        F.cv_add(res0.cv[0], res1.cv[0], cryptoContext.moduliQ, res0.cur_limbs),
        res0.cv[1],
    ]
    # res0.cv[0] = F.cv_add(
    #     res0.cv[0], res1.cv[0], cryptoContext.moduliQ, res0.cur_limbs
    # )
    return res0


def homo_mul_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    # if (isinstance(cipher, Cipher) and isinstance(plaintext, Plaintext)) :
    #     in_ct, in_pt = cipher, plaintext
    # elif (isinstance(cipher, Plaintext) and isinstance(plaintext, Cipher)):
    #     in_ct, in_pt = plaintext, cipher
    # else:
    #     raise TypeError("Invalid parameters: one must be ciphertext and the other must be plaintext.")

    assert len(cipher.cv) == 2

    if cipher.slots != plaintext.slots:
        warnings.warn(
            f"slots unequal! cipher.slots = {cipher.slots}, plaintext.slots = {plaintext.slots}",
            Warning,
        )
    if cipher.cur_limbs != plaintext.cur_limbs:
        warnings.warn(
            f"limbs unequal! cipher.cur_limbs = {cipher.cur_limbs}, plaintext.l = {plaintext.cur_limbs}, call adjust limbs function",
            Warning,
        )

    if cipher.is_ext:
        if (cipher.cur_limbs != plaintext.cur_limbs or
            cipher.noise_deg != plaintext.noise_deg or
            cipher.scaling_factor != plaintext.scaling_factor or
            cipher.is_ext != plaintext.is_ext):
            raise ValueError(f"limbs unequal! cipher.cur_limbs = {cipher.cur_limbs}, plaintext.l = {plaintext.cur_limbs}")
        moduli = cryptoContext.BsContext.QplusP_map[cipher.cur_limbs]
        mu = cryptoContext.BsContext.QmuplusPmu_map[cipher.cur_limbs]
        cv0 = F.cv_mul(cipher.cv[0], plaintext.mv, moduli, mu, cipher.cur_limbs + cryptoContext.K)
        cv1 = F.cv_mul(cipher.cv[1], plaintext.mv, moduli, mu, cipher.cur_limbs + cryptoContext.K)
        return cipher.cipher_like([cv0, cv1], scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
                                  noise_deg=cipher.noise_deg + plaintext.noise_deg)
    else:
        ctmorphed = plaintext.cipher_like(plaintext.mv) # MorphPlaintext in openfhe
        res0, res1 = _adjust_for_mult(cipher, ctmorphed, cryptoContext)

        moduli = cryptoContext.moduliQ
        mu = cryptoContext.q_mu
        cv0 = F.cv_mul(res0.cv[0], res1.cv[0], moduli, mu, res0.cur_limbs)
        cv1 = F.cv_mul(res0.cv[1], res1.cv[0], moduli, mu, res0.cur_limbs)

        return res0.cipher_like([cv0, cv1], scaling_factor=res0.scaling_factor * res1.scaling_factor,
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


