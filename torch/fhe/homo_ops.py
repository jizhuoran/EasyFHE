import math
import torch
import warnings
from .ciphertext import *
from . import functional as F
from . import hybrid_keyswitch
from .dev_tools.decorator_factory import decorator_factory


BASE_NUM_LEVELS_TO_DROP = 1  # todo: to be removed?

# drop last elem is a inplace operation now
def _drop_last_elements(ct, num_levels, cryptoContext, inplace=False):
    assert num_levels <= ct.cur_limbs and num_levels >= 0
    if not inplace:
        ct = ct.deep_copy()
    ct.cur_limbs -= num_levels
    return ct

# note: AdjustLevelsInPlace in rns-leveledshe.cpp
# mainly for "FIXEDMANUAL" case
def _adjust_levels(ct1, ct2, cryptoContext):
    ct1, ct2 = ct1.shallow_copy(), ct2.shallow_copy()
    if ct1.cur_limbs > ct2.cur_limbs:
        ct1 = _drop_last_elements(ct1, ct1.cur_limbs - ct2.cur_limbs, cryptoContext, inplace=True)
    elif ct1.cur_limbs < ct2.cur_limbs:
        ct2 = _drop_last_elements(ct2, ct2.cur_limbs - ct1.cur_limbs, cryptoContext, inplace=True)
    return ct1, ct2


def _flexauto_adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext):
    assert cipher.cur_limbs >= target_limbs
    if cipher.cur_limbs == target_limbs:
        if cipher.noise_deg < target_noise_deg:
            return _eval_mult_core(cipher, 1.0, cryptoContext)
        else:
            return cipher.shallow_copy()
    else:  # cur_limbs > target_limbs
        if cipher.noise_deg == 2 and target_noise_deg == 2:
            # if both degree 2, mul the higher to a factor, then rescale, then drop
            # interesting, ct1 actually has a noise_deg == 2, but can still do a rescale
            scf1 = cipher.scaling_factor
            scf2 = target_scaling_factor
            scf = cryptoContext.GetScalingFactorReal(cipher.cur_limbs)
            q1 = cryptoContext.GetModReduceFactor(cipher.cur_limbs - 1)
            cipher = _eval_mult_core(cipher, scf2 / scf1 * q1 / scf, cryptoContext)
            cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            if cipher.cur_limbs > target_limbs:
                cipher = _drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs,
                    cryptoContext,
                    inplace=True,
                )
            cipher.scaling_factor = target_scaling_factor
            # so the output has noise_deg2, and the same cur_limb
        elif cipher.noise_deg == 1 and target_noise_deg == 1:
            # if both degree 1, mul the higher to a factor, then drop, then rescale
            # interesting, here we can do drop first...
            scf1 = cipher.scaling_factor
            scf2 = cryptoContext.GetScalingFactorRealBig(target_limbs + 1)
            scf = cryptoContext.GetScalingFactorReal(cipher.cur_limbs)
            cipher = _eval_mult_core(cipher, scf2 / scf1 / scf, cryptoContext)
            if cipher.cur_limbs > target_limbs + 1:
                cipher = _drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs - 1,
                    cryptoContext,
                    inplace=True,
                )
            cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            cipher.scaling_factor = target_scaling_factor
            # so the output has noise_deg 1, and the same cur_limb
        elif cipher.noise_deg == 2 and target_noise_deg == 1:
            # if ct1 has degree 2, and it is just 1 more limb, do a rescale (seems this is the case the smae as fix?)
            if cipher.cur_limbs == target_limbs + 1:
                _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            else:
                # otherwise, mul the higher with scale factor, rescale, drop, rescale.
                # the last rescale is to make sure both has degree 1
                scf1 = cipher.scaling_factor
                scf2 = cryptoContext.GetScalingFactorRealBig(cryptoContext.L - (cryptoContext.L - target_limbs - 1))
                scf = cryptoContext.GetScalingFactorReal(cipher.cur_limbs)
                q1 = cryptoContext.GetModReduceFactor(cipher.cur_limbs - 1)
                cipher = _eval_mult_core(cipher, scf2 / scf1 * q1 / scf, cryptoContext)
                cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                if cipher.cur_limbs > target_limbs + 1:
                    cipher = _drop_last_elements(
                        cipher,
                        cipher.cur_limbs - target_limbs - 1,
                        cryptoContext,
                        inplace=True,
                    )
                cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                cipher.scaling_factor = target_scaling_factor
        elif cipher.noise_deg == 1 and target_noise_deg == 2:
            # if the higher has a lower degree, mul it with the factor, and then drop
            # so the output has noise_deg 2, and same cur_limb
            scf1 = cipher.scaling_factor
            scf2 = target_scaling_factor
            scf = cryptoContext.GetScalingFactorReal(cipher.cur_limbs)
            cipher = _eval_mult_core(cipher, scf2 / scf1 / scf, cryptoContext)
            cipher = _drop_last_elements(cipher, cipher.cur_limbs - target_limbs, cryptoContext, inplace=True)
            cipher.scaling_factor = scf2
        else:
            print("noise_deg", cipher.noise_deg, target_noise_deg)
            raise ValueError

    return cipher


def _fixauto_adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext):
    assert cipher.cur_limbs >= target_limbs
    if cipher.cur_limbs == target_limbs:
        if cipher.noise_deg < target_noise_deg:
            return _eval_mult_core(cipher, 1.0, cryptoContext)
        else:
            return cipher.shallow_copy()
    else:  # cur_limbs > target_limbs
        if cipher.noise_deg == 2 and target_noise_deg == 2:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            if cipher.cur_limbs > target_limbs:
                cipher = _drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs,
                    cryptoContext,
                    inplace=False,
                )
        elif cipher.noise_deg == 1 and target_noise_deg == 1:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            if cipher.cur_limbs > target_limbs + 1:
                cipher = _drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs - 1,
                    cryptoContext,
                    inplace=True,
                )
            cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        elif cipher.noise_deg == 2 and target_noise_deg == 1:
            if cipher.cur_limbs == target_limbs + 1:
                _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            else:
                cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
                cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                if cipher.cur_limbs > target_limbs + 1:
                    cipher = _drop_last_elements(
                        cipher,
                        cipher.cur_limbs - target_limbs - 1,
                        cryptoContext,
                        inplace=True,
                    )
                cipher = _homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        elif cipher.noise_deg == 1 and target_noise_deg == 2:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            cipher = _drop_last_elements(cipher, cipher.cur_limbs - target_limbs, cryptoContext, inplace=True)
        else:
            print("noise_deg", cipher.noise_deg, target_noise_deg)
            raise ValueError
    return cipher

def _fixmanual_adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext):
    assert cipher.noise_deg == 1 and target_noise_deg == 1
    return _drop_last_elements(cipher, cipher.cur_limbs - target_limbs, cryptoContext, inplace=False)

def _adjust_levels_and_depth(ct1, ct2, cryptoContext):
    if ct1.cur_limbs > ct2.cur_limbs:
        target_limbs = ct2.cur_limbs
        target_noise_deg = ct2.noise_deg
        target_scaling_factor = ct2.scaling_factor
        # print("case1", target_limbs, target_noise_deg, target_scaling_factor)
    elif ct1.cur_limbs < ct2.cur_limbs:
        target_limbs = ct1.cur_limbs
        target_noise_deg = ct1.noise_deg
        target_scaling_factor = ct1.scaling_factor
        # print("case2", target_limbs, target_noise_deg, target_scaling_factor)
    else:
        target_limbs = ct1.cur_limbs
        target_noise_deg = max(ct1.noise_deg, ct2.noise_deg)
        target_scaling_factor = None
        # print("case3", target_limbs, target_noise_deg, target_scaling_factor)
    return _adjust_to(ct1, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext), _adjust_to(
        ct2, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
    )

# AdjustForAddOrSubInPlace in rns-leveledshe.cpp
def _adjust_for_add_or_sub(in0, in1, cryptoContext):
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        assert in0.noise_deg == in1.noise_deg
        return _adjust_levels(in0, in1, cryptoContext)
    else:
        return _adjust_levels_and_depth(in0, in1, cryptoContext)


def _adjust_for_mult(ct1: Cipher, ct2: Cipher, cryptoContext):
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        rct1, rct2 = _adjust_levels(ct1, ct2, cryptoContext)
    else:
        # inline `AdjustLevelsAndDepthToOneInPlace` in ckksrns-leveledshe.cpp as following
        rct1, rct2 = _adjust_levels_and_depth(ct1, ct2, cryptoContext)
        if rct1.noise_deg == 2:
            rct1 = _homo_rescale_internal(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        if rct2.noise_deg == 2:  # if rct1's noise_deg is 1, so is rct2
            rct2 = _homo_rescale_internal(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return rct1, rct2


from enum import Enum


class LargeScalingFactorConstants(Enum):
    MAX_BITS_IN_WORD = 61
    MAX_LOG_STEP = 60


# CRTMult in ckkspackedencoding.cpp
def _crt_mult(xs, ys, mods):
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
            crt_approx = _crt_mult(crt_approx, crt_sf, cryptoContext.moduliQ_scalar)
            log_approx -= log_step

        crt_constant = _crt_mult(crt_constant, crt_approx, cryptoContext.moduliQ_scalar)

    # Handle FLEXIBLEAUTOEXT mode at level 0, we don't use the depth to calculate the scaling factor,
    # so we return the value before taking the depth into account.
    if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT" and cur_limbs == cryptoContext.L:
        return crt_constant

    # Final scaling factor adjustments
    int_sc_factor = int(sc_factor + 0.5)
    crt_sc_factor = cur_limbs * [int_sc_factor]

    for i in range(1, noise_deg):
        crt_constant = _crt_mult(crt_constant, crt_sc_factor, cryptoContext.moduliQ_scalar)

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
        log_valid = log_sf if log_sf <= MAX_BITS_IN_WORD_LOCAL else MAX_BITS_IN_WORD_LOCAL
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
        log_step = log_approx if log_approx <= LargeScalingFactorConstants.MAX_LOG_STEP.value else LargeScalingFactorConstants.MAX_LOG_STEP.value
        int_step = 1 << log_step
        crt_approx = cur_limbs * [int_step]
        log_approx -= log_step

        while log_approx > 0:
            log_step = log_approx if log_approx <= LargeScalingFactorConstants.MAX_LOG_STEP.value else LargeScalingFactorConstants.MAX_LOG_STEP.value
            int_step = 1 << log_step
            crt_sf = cur_limbs * [int_step]
            crt_approx = _crt_mult(crt_approx, crt_sf, cryptoContext.moduliQ_scalar)
            log_approx -= log_step
        factors = _crt_mult(factors, crt_approx, cryptoContext.moduliQ_scalar)

    return factors


# note: EvalMultCoreInPlace in ckksrns-leveledshe.cpp
def _eval_mult_core(in0, cnst, cryptoContext):
    factors = _get_element_for_eval_mult(cnst, in0.cur_limbs, cryptoContext)
    return _cipher_mul_scalar_double(in0, factors, cryptoContext)


# todo: only support in `FIXEDMANUAL` mode, or `_adjust_levels_and_depth` function.
# should not be used directly in other rescale modes!!! except when openfhe directly used it
# todo: write homo_level_reduce


def _cipher_add(in0, in1, cryptoContext):
    cv = [F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
    return in0.cipher_like(cv)


def _cipher_add_ext(in0, in1, cryptoContext):
    cv = [
        F.cv_add(
            cv0,
            cv1,
            cryptoContext.QplusP_map[in0.cur_limbs],
            in0.cur_limbs + cryptoContext.K,
        )
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


def _cipher_sub(in0, in1, cryptoContext):
    cv = [F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
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
    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return in0.cipher_like(
        [bx, ax, axax],
        scaling_factor=in0.scaling_factor * scFactor,
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
    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return in0.cipher_like(
        [bx, ax, axax],
        scaling_factor=in0.scaling_factor * scFactor,
        noise_deg=in0.noise_deg + in0.noise_deg,
    )


def _cipher_add_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


def _cipher_sub_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


# todo: used only in `homo_mul_scalar_int`, therefore the scaling factor and noise_deg remain unchanged
# todo: if used for `homo_mul_scalar_double`, the scaling factor and noise_deg should be changed
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


def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)

def _adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext):
    assert (cipher.cur_limbs - cipher.noise_deg) >= (target_limbs - target_noise_deg)
    if cryptoContext.rescaleTech == "FLEXIBLEAUTO":
        return _flexauto_adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext)
    elif cryptoContext.rescaleTech == "FIXEDAUTO":
        return _fixauto_adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext)
    elif cryptoContext.rescaleTech == "FIXEDMANUAL":
        return _fixmanual_adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext)
    else:
        raise ValueError 


@decorator_factory
def adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext):
    return _adjust_to(cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext)




def _cipher_automorphism(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    limbs = in0.cur_limbs if in0.is_ext == False else in0.cur_limbs + cryptoContext.K
    cv = [F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext) for cv in in0.cv]
    return in0.cipher_like(cv)

# todo: input len of in0.cv could be 1
@decorator_factory
def cipher_automorphism(in0, index, cryptoContext):
    return _cipher_automorphism(in0, index, cryptoContext)


@decorator_factory
def homo_add(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_add_ext(in0, in1, cryptoContext)
    else:
        return _cipher_add(in0, in1, cryptoContext)


@decorator_factory
def homo_sub(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)


@decorator_factory
def homo_rescale(ct, levels, cryptoContext):  # todo: add force_rescale flag in user api for other rescaleTech?
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        return _homo_rescale_internal(ct, levels, cryptoContext)
    else:
        return ct.deep_copy()


def _homo_rescale_internal(ct, levels, cryptoContext):
    assert levels == 1 or levels == 0 and "Only support these two cases"
    assert ct.cur_limbs-levels > 0, "there aren't enough limbs to be rescaled"
    if levels == 0:
        return ct.deep_copy()

    def _rescale_n_times(cv, levels):
        for l in range(levels):
            cv = F.cv_drop_last_element_and_scale(cv, ct.cur_limbs, l, cryptoContext)
        return cv

    res_cv = [_rescale_n_times(_cv, levels) for _cv in ct.cv]

    scFactor = ct.scaling_factor
    for l in range(levels):
        modReduceFactor = cryptoContext.GetModReduceFactor(
            ct.cur_limbs - 1 - l
        )  # corresponding to openfhe: (sizeQl -1 -i), we need to use the value of input ct
        scFactor = scFactor / modReduceFactor

    return ct.cipher_like(
        res_cv,
        cur_limbs=ct.cur_limbs - levels,
        scaling_factor=scFactor,
        noise_deg=ct.noise_deg - levels,
    )

@decorator_factory
def force_rescale(ct, levels, cryptoContext):
    return _homo_rescale_internal(ct, levels, cryptoContext)

@decorator_factory
def homo_mul(in0, in1, cryptoContext):
    # note: AdjustForMultInPlace in rns-leveledshe.cpp
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


@decorator_factory
def homo_square(in0, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg != 1:
        in0 = _homo_rescale_internal(in0, 1, cryptoContext)
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


@decorator_factory
def homo_add_scalar_double(in0, cnst, cryptoContext, precomp=None):
    tmpr = _get_element_for_eval_add_or_sub(math.fabs(cnst), in0.cur_limbs, in0.noise_deg, cryptoContext)
    if cnst < 0:
        return _cipher_sub_scalar(in0, tmpr, cryptoContext)
    else:
        return _cipher_add_scalar(in0, tmpr, cryptoContext)


def homo_add_scalar_int(in0, scalar, cryptoContext):
    return _cipher_add_scalar(in0, scalar, cryptoContext)


# note: corresponds to MultByIntegerInPlace in openfhe, the datatype of scalar in openfhe is `uint64_t`
# fixme: do we accept scalar<0?
@decorator_factory
def homo_mul_scalar_int(in0, scalar, cryptoContext):
    res = _cipher_mul_scalar_int(in0, abs(scalar), cryptoContext)
    if scalar < 0:
        res = _cipher_neg(res, cryptoContext)
    return res


# note: EvalMultInPlace in ckksrns-leveledshe.cpp
@decorator_factory
def homo_mul_scalar_double(in0, cnst, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg == 2:
        in0 = _homo_rescale_internal(in0, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return _eval_mult_core(in0, cnst, cryptoContext)


@decorator_factory
def homo_rotate(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.get_rotation_key(norm_index)
    res = in0.cipher_like(F.cv_keyswitch(in0.cv[1], in0.cur_limbs, swk[0], swk[1], cryptoContext))

    res.cv[0] = F.cv_add(in0.cv[0], res.cv[0], cryptoContext.moduliQ, in0.cur_limbs)

    res = _cipher_automorphism(res, index, cryptoContext)

    return res


@decorator_factory
def eval_fast_rotate(digits, cipher, index, need_KS_add, need_moddown, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    result = hybrid_keyswitch._mult_rot_key_and_sum_ext(digits, index, cryptoContext)

    if need_KS_add:
        # if need_moddown ==False, we don't care cipher.is_ext
        # if need_moddown ==True, cipher.is_ext should be False
        assert not(need_moddown==True and cipher.is_ext==True), "contradict input"
        if cipher.is_ext ==True:
            result.cv[0] = F.cv_add(
                result.cv[0],
                cipher.cv[0],
                cryptoContext.QplusP_map[cipher.cur_limbs],
                cipher.cur_limbs + cryptoContext.K,
            )
        else:
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
                )  # PModUp

            # post add after ks
            # if need_moddown = False, operate sumMult.cv[0][:curr_limbs], and sumMult.cv[0][curr_limbs+1:] remain unchanged,
            # so the `inplace` can't be removed trivially
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


@decorator_factory
def homo_add_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    # res0 = cipher.deep_copy()
    ctmorphed = plaintext.cipher_like(plaintext.cv)  # MorphPlaintext in openfhe
    res0, res1 = _adjust_for_add_or_sub(cipher, ctmorphed, cryptoContext)
    res0.cv = [
        F.cv_add(res0.cv[0], res1.cv[0], cryptoContext.moduliQ, res0.cur_limbs),
        res0.cv[1],
    ]
    # res0.cv[0] = F.cv_add(
    #     res0.cv[0], res1.cv[0], cryptoContext.moduliQ, res0.cur_limbs
    # )
    return res0


@decorator_factory
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
        if (
            cipher.cur_limbs != plaintext.cur_limbs
            or cipher.noise_deg != plaintext.noise_deg
            or cipher.scaling_factor != plaintext.scaling_factor
            or cipher.is_ext != plaintext.is_ext
        ):
            raise ValueError(
                f"Unequal values! Cipher and plaintext have mismatched properties:\n"
                f"  cipher.cur_limbs = {cipher.cur_limbs}, plaintext.cur_limbs = {plaintext.cur_limbs}\n"
                f"  cipher.noise_deg = {cipher.noise_deg}, plaintext.noise_deg = {plaintext.noise_deg}\n"
                f"  cipher.scaling_factor = {cipher.scaling_factor}, plaintext.scaling_factor = {plaintext.scaling_factor}\n"
                f"  cipher.is_ext = {cipher.is_ext}, plaintext.is_ext = {plaintext.is_ext}"
            )
        moduli = cryptoContext.QplusP_map[cipher.cur_limbs]
        mu = cryptoContext.QmuplusPmu_map[cipher.cur_limbs]
        cv0 = F.cv_mul(cipher.cv[0], plaintext.cv[0], moduli, mu, cipher.cur_limbs + cryptoContext.K)
        cv1 = F.cv_mul(cipher.cv[1], plaintext.cv[0], moduli, mu, cipher.cur_limbs + cryptoContext.K)
        return cipher.cipher_like(
            [cv0, cv1],
            scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
            noise_deg=cipher.noise_deg + plaintext.noise_deg,
        )
    else:
        ctmorphed = plaintext.cipher_like(plaintext.cv)  # MorphPlaintext in openfhe
        res0, res1 = _adjust_for_mult(cipher, ctmorphed, cryptoContext)

        moduli = cryptoContext.moduliQ
        mu = cryptoContext.q_mu
        cv0 = F.cv_mul(res0.cv[0], res1.cv[0], moduli, mu, res0.cur_limbs)
        cv1 = F.cv_mul(res0.cv[1], res1.cv[0], moduli, mu, res0.cur_limbs)

        return res0.cipher_like(
            [cv0, cv1],
            scaling_factor=res0.scaling_factor * res1.scaling_factor,
            noise_deg=res0.noise_deg + res1.noise_deg,
        )


@decorator_factory
def extract_cv(cipher: Cipher, index, cryptoContext, append_zeros=False):
    assert index == 0 or index == 1, "index must be 0 or 1"
    if append_zeros:
        if index == 0:
            return cipher.cipher_like([cipher.cv[0], torch.zeros_like(cipher.cv[0])])
        else:
            return cipher.cipher_like([torch.zeros_like(cipher.cv[1]), cipher.cv[1]])
    else:
        return cipher.cipher_like([cipher.cv[index]])


import numpy as np

MAX_BITS_IN_WORD = 61
MAX_64BIT_VALUE = (1 << 63) - (1 << 9) - 1  # openfhetodo: the var must be renamed


def _fft_special_inv(vals, M, rotGroup, ksiPows):

    def _bit_reverse(vals):
        size = len(vals)
        vals = np.array(vals, dtype=np.complex128)  # 转为 numpy 复数数组
        j = 0
        for i in range(1, size):
            bit = size >> 1
            while j >= bit:
                j -= bit
                bit >>= 1
            j += bit
            if i < j:
                vals[i], vals[j] = vals[j], vals[i]  # 交换复数
        return vals

    vals_size = len(vals)

    # FFT特定的操作
    len_size = vals_size
    while len_size >= 1:
        len_h = len_size >> 1
        len_q = len_size << 2
        gap = M // len_q

        for i in range(0, vals_size, len_size):
            for j in range(len_h):
                idx = (len_q - (rotGroup[j] % len_q)) * gap
                u = vals[i + j] + vals[i + j + len_h]
                v = vals[i + j] - vals[i + j + len_h]
                v *= ksiPows[idx]
                vals[i + j] = u
                vals[i + j + len_h] = v
        len_size >>= 1

    vals = _bit_reverse(vals)

    for i in range(vals_size):
        vals[i] /= vals_size
    return vals


def dump_encode_middle(
    inverse,
    slots,
    scaling_factor,
    cryptocontext,
):
    inverse_complex = np.array([complex(v.real, 0.0) for v in inverse])

    # Resize the inverse to fit the slot size.
    # note that default: slots value should be greater than size of input data list x
    inverse_complex = np.pad(
        inverse_complex,
        pad_width=(0, slots - len(inverse)),
        mode="constant",
        constant_values=complex(0.0, 0.0),
    )
    arr = cryptocontext.encode_params_ksiPows.cpu().numpy()
    complex_arr = arr[0::2] + arr[1::2] * 1j
    inverse_complex = _fft_special_inv(
        inverse_complex,
        cryptocontext.M,
        cryptocontext.encode_params_rotGroup.cpu().numpy(),
        complex_arr,
    )
    logc = 0
    for i in range(slots):
        inverse[i] *= scaling_factor
        if inverse[i].real != 0:
            logci = int(math.ceil(math.log2(abs(inverse[i].real))))
            logc = max(logc, logci)
        if inverse[i].imag != 0:
            logci = int(math.ceil(math.log2(abs(inverse[i].imag))))
            logc = max(logc, logci)

    if logc < 0:
        raise ValueError("Too small scaling factor")

    log_valid = min(logc, MAX_BITS_IN_WORD)
    log_approx = logc - log_valid
    inverse_array = np.array(inverse_complex, dtype=np.complex128).view(np.float64).tolist()
    return inverse_array, log_approx

def pre_encode(x, slots):
    import cmath

    inverse = x

    N = 1 << 16
    M = N << 1
    Nh = N >> 1

    # compute encode params
    M_PI = 3.14159265358979323846
    fivePows = 1
    encode_params_ksiPows = []
    encode_params_rotGroup = []
    for i in range(Nh):
        encode_params_rotGroup.append(fivePows)
        fivePows = (fivePows * 5) % M

    # m_ksiPows stores the complex roots of unity
    for j in range(M):
        angle = 2.0 * M_PI * j / M
        encode_params_ksiPows.append(cmath.exp(1j * angle))
    encode_params_ksiPows.append(encode_params_ksiPows[0])

    encode_params_ksiPows = np.array(encode_params_ksiPows, dtype=np.complex128).view(np.float64).tolist()
    encode_params_rotGroup = np.array(encode_params_rotGroup)

    if slots < len(inverse):
        raise ValueError(f"The number of slots [{slots}] is less than the size of data [{len(inverse)}]")

    # Clears all imaginary values as CKKS for complex numbers
    inverse_complex = np.array([complex(v.real, 0.0) for v in inverse])

    # Resize the inverse to fit the slot size.
    # note that default: slots value should be greater than size of input data list x
    inverse_complex = np.pad(
        inverse_complex,
        pad_width=(0, slots - len(inverse)),
        mode="constant",
        constant_values=complex(0.0, 0.0),
    )
    arr = np.array(encode_params_ksiPows, dtype=np.float64)
    complex_arr = arr[0::2] + arr[1::2] * 1j
    inverse_complex = _fft_special_inv(
        inverse_complex,
        M,
        np.array(encode_params_rotGroup, dtype=np.int32),
        complex_arr,
    )
    inverse_array = np.array(inverse_complex, dtype=np.complex128).view(np.float64)
    max_encoded_value = np.max(np.abs(inverse_array))

    encoded_val = PreEncodeValues(
        np.pad(
            x,
            pad_width=(0, slots - len(x)),
            mode="constant",
            constant_values=0.0,
        ),
        slots,
        inverse_array,
        max_encoded_value,
    )
    return encoded_val


@decorator_factory
def encode(
    x,
    name,
    level,
    slots,
    cryptoContext
):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        middle_value = pre_encode(x, slots)
        middle_value.encoded_values = torch.tensor(middle_value.encoded_values, dtype=torch.double, device="cuda")
    elif isinstance(x, PreEncodeValues):
        assert slots == x.slots
        middle_value = x
    elif isinstance(x, Plaintext):
        return x
    else:
        raise ValueError("Invalid input type")
        
    cur_limbs = cryptoContext.L - level

    if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT":
        scaling_factor = cryptoContext.GetScalingFactorRealBig(cur_limbs)
    else:
        scaling_factor = cryptoContext.GetScalingFactorReal(cur_limbs)

    assert middle_value.max_encoded_value < 1e-20 or math.log2(int(middle_value.max_encoded_value * scaling_factor)) < 61 #MAX_BITS_IN_WORD

    pt_encode = torch.encode(
        input=middle_value.encoded_values,
        N=cryptoContext.N,
        cur_limbs=cur_limbs,
        slots=slots,
        scaling_factor=scaling_factor,
        primes=cryptoContext.primes,
        barret_ratio=cryptoContext.barret_ratio,
        barret_k=cryptoContext.barret_k,
        power_of_roots_shoup=cryptoContext.power_of_roots_shoup,
        power_of_roots=cryptoContext.power_of_roots
    )

    gpufhe_cipher = Plaintext([pt_encode], pt_encode.shape[0], scaling_factor, 1, slots, False)
    if cryptoContext.config.PTX_TWIN:
        gpufhe_cipher.ptx_twin = np.array(x.tolist() + [0] * (slots - len(x)))
    return gpufhe_cipher


# def encode(
#     x,
#     scale_deg,
#     level,
#     slots,
#     use_gpu_fft,
#     cryptoContext,
#     use_middle=False,
#     inverse_internal=None,
#     log_approx=None,
# ):

#     def _ptx_encode_cuda(
#         x,
#         slots,
#         type_flag,
#         scaling_factor,
#         cur_limbs,
#         noise_scale_deg,
#         use_gpu_fft,
#         cryptocontext,
#     ):
#         inverse = x
#         pt_encode = []

#         if slots < len(inverse):
#             raise ValueError(f"The number of slots [{slots}] is less than the size of data [{len(inverse)}]")

#         if not use_gpu_fft:
#             # Clears all imaginary values as CKKS for complex numbers
#             inverse_complex = np.array([complex(v.real, 0.0) for v in inverse])

#             # Resize the inverse to fit the slot size.
#             # note that default: slots value should be greater than size of input data list x
#             inverse_complex = np.pad(
#                 inverse_complex,
#                 pad_width=(0, slots - len(inverse)),
#                 mode="constant",
#                 constant_values=complex(0.0, 0.0),
#             )
#             arr = cryptocontext.encode_params_ksiPows.cpu().numpy()
#             complex_arr = arr[0::2] + arr[1::2] * 1j
#             inverse_complex = _fft_special_inv(
#                 inverse_complex,
#                 cryptocontext.M,
#                 cryptocontext.encode_params_rotGroup.cpu().numpy(),
#                 complex_arr,
#             )
#             inverse_array = np.array(inverse_complex, dtype=np.complex128).view(np.float64).tolist()
#             inverse_internal = torch.tensor(inverse_array, dtype=torch.double, device="cuda")
#             inverse = torch.tensor(inverse, device="cuda")
#         else:
#             inverse_internal = cryptocontext.encode_inverse

#         pt_encode = torch.encode(
#             inverse=inverse,
#             inverse_internal=inverse_internal,
#             temp=cryptocontext.encode_temp,
#             primes=cryptocontext.primes,
#             precompute_rotgroups=cryptocontext.encode_params_rotGroup,
#             precompute_ksipows=cryptocontext.encode_params_ksiPows,
#             M=cryptocontext.M,
#             N=cryptocontext.N,
#             cur_limbs=cur_limbs,
#             slots=slots,
#             noise_scale_deg=noise_scale_deg,
#             scaling_factor=scaling_factor,
#             power_of_roots_shoup=cryptocontext.power_of_roots_shoup,
#             power_of_roots=cryptocontext.power_of_roots,
#             use_fft=use_gpu_fft,
#         )

#         return pt_encode

#     def _ptx_encode_middle_cuda(
#         inverse_internal,
#         slots,
#         type_flag,
#         scaling_factor,
#         log_approx,
#         cur_limbs,
#         noise_scale_deg,
#         cryptocontext,
#     ):
#         pt_encode = torch.encode_middle(
#             inverse_internal=inverse_internal,
#             primes=cryptocontext.primes,
#             N=cryptocontext.N,
#             cur_limbs=cur_limbs,
#             slots=slots,
#             noise_scale_deg=noise_scale_deg,
#             scaling_factor=scaling_factor,
#             log_approx=log_approx,
#             power_of_roots_shoup=cryptocontext.power_of_roots_shoup,
#             power_of_roots=cryptocontext.power_of_roots,
#         )
#         return pt_encode

#     cur_limb = cryptoContext.L - level
#     if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT":
#         scFact = cryptoContext.GetScalingFactorRealBig(cur_limb)
#         # In FLEXIBLEAUTOEXT mode at level 0, we don't use the noiseScaleDeg
#         # in our encoding function, so we set it to 1 to make sure it
#         # has no effect on the encoding.
#         assert scale_deg == 1
#     else:
#         scFact = cryptoContext.GetScalingFactorReal(cur_limb)

#     if not use_middle:
#         encoded_vector_dcrt_elements_cuda = _ptx_encode_cuda(
#             x,
#             slots,
#             "IsDCRTPoly",
#             scFact,
#             cur_limb,
#             scale_deg,
#             use_gpu_fft,
#             cryptoContext,
#         )
#         mv = [encoded_vector_dcrt_elements_cuda]
#         gpufhe_cipher = Plaintext(mv, mv[0].shape[0], scFact, scale_deg, slots, False)
#         if cryptoContext.config.PTX_TWIN:
#             gpufhe_cipher.ptx_twin = np.array(x.tolist() + [0] * (slots - len(x)))
#         return gpufhe_cipher
#     else:
#         if inverse_internal != None and log_approx != None:
#             encoded_vector_dcrt_elements_cuda = _ptx_encode_middle_cuda(
#                 inverse_internal,
#                 slots,
#                 "IsDCRTPoly",
#                 scFact,
#                 log_approx,
#                 cur_limb,
#                 scale_deg,
#                 cryptoContext,
#             )
#             mv = [encoded_vector_dcrt_elements_cuda]
#             gpufhe_cipher = Plaintext(mv, mv[0].shape[0], scFact, scale_deg, slots, False)
#             if cryptoContext.config.PTX_TWIN:
#                 gpufhe_cipher.ptx_twin = np.array(x + [0] * (slots - len(x)))
#             return gpufhe_cipher
#         else:
#             inverse_internal = torch.encode_fft(
#                 inverse=x,
#                 precompute_rotgroups=cryptoContext.encode_params_rotGroup,
#                 precompute_ksipows=cryptoContext.encode_params_ksiPows,
#                 M=cryptoContext.M,
#                 slots=slots,
#             )
#             log_approx = torch.encode_log_approx(
#                 inverse_internal=inverse_internal,
#                 slots=slots,
#                 cur_limbs=cur_limb,
#                 scaling_factor=scFact,
#             )
#             log_approx = int(log_approx.cpu()[0])

#             encoded_vector_dcrt_elements_cuda = _ptx_encode_middle_cuda(
#                 inverse_internal,
#                 slots,
#                 "IsDCRTPoly",
#                 scFact,
#                 log_approx,
#                 cur_limb,
#                 scale_deg,
#                 cryptoContext,
#             )
#             mv = [encoded_vector_dcrt_elements_cuda]
#             gpufhe_cipher = Plaintext(mv, mv[0].shape[0], scFact, scale_deg, slots, False)
#             if cryptoContext.config.PTX_TWIN:
#                 gpufhe_cipher.ptx_twin = np.array(x + [0] * (slots - len(x)))
#             return (
#                 inverse_internal,
#                 log_approx,
#                 gpufhe_cipher
#             )


import math
from math import log2, pi

def select_layers(log_slots, budget):
    layers = int(math.ceil(log_slots / budget))
    rows = int(log_slots // layers)
    rem = log_slots % layers

    dim = rows
    if rem != 0:
        dim = rows + 1

    # Ensure dim <= budget
    if dim < budget:
        layers -= 1
        rows = log_slots // layers
        rem = log_slots - rows * layers
        dim = rows

        if rem != 0:
            dim = rows + 1

        # Ensure dim >= budget
        while dim != budget:
            rows -= 1
            rem = log_slots - rows * layers
            dim = rows
            if rem != 0:
                dim = rows + 1

    return [int(layers), int(rows), int(rem)]

def get_collapsed_fft_params(slots, level_budget, dim1):
    log_slots = math.floor(math.log2(slots))
    # Even for the case of a single slot, we need one level for rescaling
    if log_slots == 0:
        log_slots = 1

    dims = select_layers(log_slots, level_budget)
    layers_collapse = dims[0]
    rem_collapse = dims[2]

    flag_rem = rem_collapse != 0

    num_rotations = (1 << (layers_collapse + 1)) - 1
    num_rotations_rem = (1 << (rem_collapse + 1)) - 1

    # Compute baby-step (b) and giant-step (g) for collapsed layers
    if dim1 == 0 or dim1 > num_rotations:
        if num_rotations > 7:
            g = 1 << (layers_collapse // 2 + 2)
        else:
            g = 1 << (layers_collapse // 2 + 1)
    else:
        g = dim1
    b = (num_rotations + 1) // g

    b_rem = 0
    g_rem = 0
    if flag_rem:
        if num_rotations_rem > 7:
            g_rem = 1 << (rem_collapse // 2 + 2)
        else:
            g_rem = 1 << (rem_collapse // 2 + 1)
        b_rem = (num_rotations_rem + 1) // g_rem

    # Return the parameters
    return [
        int(level_budget), layers_collapse, rem_collapse, int(num_rotations), b, g,
        int(num_rotations_rem), b_rem, g_rem
    ]


def coeff_encoding_one_level(pows, rot_group, flag_i):
    dim = len(pows) - 1
    slots = len(rot_group)

    # Initialize the coefficient matrix
    # coeff = [[np.zeros(slots, dtype=np.complex128) for _ in range(3 * int(log2(slots)))]]
    coeff = [[0j] * slots for _ in range(int(3 * math.log2(slots)))]

    m = slots
    while m > 1:
        s = int(log2(m)) - 1

        for k in range(0, slots, m):
            lenh = m >> 1
            lenq = m << 2

            for j in range(lenh):
                j_twiddle = (lenq - (rot_group[j] % lenq)) * (dim // lenq)

                if flag_i and (m == 2):
                    w = np.exp(-1j * pi / 2) * pows[j_twiddle]
                    coeff[s + int(log2(slots))][j + k] = np.exp(-1j * pi / 2)  # not shifted
                    coeff[s + 2 * int(log2(slots))][j + k] = np.exp(-1j * pi / 2)  # shifted left
                    coeff[s + int(log2(slots))][j + k + lenh] = -w  # not shifted
                    coeff[s][j + k + lenh] = w  # shifted right
                else:
                    w = pows[j_twiddle]
                    coeff[s + int(log2(slots))][j + k] = 1  # not shifted
                    coeff[s + 2 * int(log2(slots))][j + k] = 1  # shifted left
                    coeff[s + int(log2(slots))][j + k + lenh] = -w  # not shifted
                    coeff[s][j + k + lenh] = w  # shifted right
        m >>= 1

    return coeff


def reduce_rotation(index, slots):
    islots = int(slots)

    # If slots is a power of 2
    if (slots & (slots - 1)) == 0:
        n = int(math.log2(slots))
        if index >= 0:
            return index - ((index >> n) << n)
        return index + islots + ((abs(index) >> n) << n)

    return (islots + index % islots) % islots

def coeff_encoding_collapse(pows, rot_group, level_budget, flag_i):
    slots = len(rot_group)

    # Compute how many layers are collapsed in each level from the budget
    dims = select_layers(log2(slots), level_budget)
    layers_collapse = dims[0]
    rem_collapse = dims[2]

    dim_collapse = level_budget
    stop = 0
    flag_rem = 0

    if rem_collapse == 0:
        stop = -1
        flag_rem = 0
    else:
        stop = 0
        flag_rem = 1

    num_rotations = (1 << (layers_collapse + 1)) - 1
    num_rotations_rem = (1 << (rem_collapse + 1)) - 1

    # Compute the coefficients for encoding for the given level budget
    coeff1 = coeff_encoding_one_level(pows, rot_group, flag_i)

    # Coeff stores the coefficients for the given budget of levels
    coeff = []
    for i in range(dim_collapse):
        if flag_rem:
            if i >= 1:
                # After remainder
                coeff.append([[0j] * slots for _ in range(num_rotations)])
            else:
                # Remainder corresponds to the first index in encoding and to the last one in decoding
                coeff.append([[0j] * slots for _ in range(num_rotations_rem)])
        else:
            coeff.append([[0j] * slots for _ in range(num_rotations)])

    for s in range(dim_collapse - 1, stop, -1):
        top = int(log2(slots)) - (dim_collapse - 1 - s) * layers_collapse - 1

        for l in range(layers_collapse):
            if l == 0:
                coeff[s][0] = coeff1[top]
                coeff[s][1] = coeff1[top + int(log2(slots))]
                coeff[s][2] = coeff1[top + 2 * int(log2(slots))]
            else:
                temp = coeff[s]
                zeros = [[0.0] * slots for _ in range(num_rotations)]
                coeff[s] = zeros
                t = 0

                for u in range((1 << (l + 1)) - 1):
                    for k in range(slots):
                        coeff[s][u + t][k] += coeff1[top - l][k] * temp[u][reduce_rotation(k - (1 << (top - l)), slots)]
                        coeff[s][u + t + 1][k] += coeff1[top - l + int(log2(slots))][k] * temp[u][k]
                        coeff[s][u + t + 2][k] += coeff1[top - l + 2 * int(log2(slots))][k] * temp[u][reduce_rotation(k + (1 << (top - l)), slots)]
                    t += 1

    if flag_rem:
        s = 0
        top = int(log2(slots)) - (dim_collapse - 1 - s) * layers_collapse - 1

        for l in range(rem_collapse):
            if l == 0:
                coeff[s][0] = coeff1[top]
                coeff[s][1] = coeff1[top + int(log2(slots))]
                coeff[s][2] = coeff1[top + 2 * int(log2(slots))]
            else:
                temp = coeff[s]
                zeros = [[0j] * slots for _ in range(num_rotations_rem)]
                coeff[s] = zeros
                t = 0

                for u in range((1 << (l + 1)) - 1):
                    for k in range(slots):
                        coeff[s][u + t][k] += coeff1[top - l][k] * temp[u][reduce_rotation(k - (1 << (top - l)), slots)]
                        coeff[s][u + t + 1][k] += coeff1[top - l + int(log2(slots))][k] * temp[u][k]
                        coeff[s][u + t + 2][k] += coeff1[top - l + 2 * int(log2(slots))][k] * temp[u][reduce_rotation(k + (1 << (top - l)), slots)]
                    t += 1

    return coeff


def rotate(a, index):
    slots = len(a)
    result = np.zeros(slots, dtype=np.complex128)

    if index < 0 or index > slots:
        index = reduce_rotation(index, slots)

    if index == 0:
        result = np.array(a, dtype=np.complex128)
    else:
        # Two cases: i + index <= slots and i + index > slots
        result[:slots - index] = a[index:]
        result[slots - index:] = a[:index]

    return result


def encode_bsMatrix(
    x,
    name,
    level,
    slots,
    cryptoContext
):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        middle_value = pre_encode(x, slots)
        middle_value.encoded_values = torch.tensor(middle_value.encoded_values, dtype=torch.double, device="cuda")
    elif isinstance(x, PreEncodeValues):
        assert slots == x.slots
        middle_value = x
    elif isinstance(x, Plaintext):
        return x
    else:
        raise ValueError("Invalid input type")

    cur_limbs = cryptoContext.L - level

    if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT":
        scaling_factor = cryptoContext.GetScalingFactorRealBig(cur_limbs)
    else:
        scaling_factor = cryptoContext.GetScalingFactorReal(cur_limbs)

    assert middle_value.max_encoded_value < 1e-20 or math.log2(
        int(middle_value.max_encoded_value * scaling_factor)) < 61  # MAX_BITS_IN_WORD

    pt_encode = torch.encode(
        input=middle_value.encoded_values,
        N=cryptoContext.N,
        cur_limbs=cur_limbs,
        slots=slots,
        scaling_factor=scaling_factor,
        primes=cryptoContext.encode_params["primes"],
        barret_ratio=cryptoContext.encode_params["barret_ratio"],
        barret_k=cryptoContext.encode_params["barret_k"],
        power_of_roots_shoup=cryptoContext.encode_params["power_of_roots_shoup"],
        power_of_roots=cryptoContext.encode_params["power_of_roots"]
    )

    gpufhe_cipher = Plaintext([pt_encode], pt_encode.shape[0], scaling_factor, 1, slots, False)
    if cryptoContext.config.PTX_TWIN:
        gpufhe_cipher.ptx_twin = np.array(x.tolist() + [0] * (slots - len(x)))
    return gpufhe_cipher


def eval_coeffs_to_slots_precompute(scale, lRemain, cryptoContext):
    import copy
    # copied from pre_encode
    import cmath

    N = cryptoContext.N
    M = cryptoContext.M
    Nh = N >> 1
    precom = cryptoContext.BsContext_map[str(int(math.log2(Nh)))] #TODO: SUPPORT different slots, Nh should be the value of slots
    # compute encode params
    M_PI = 3.14159265358979323846
    fivePows = 1

    encode_params_ksiPows = []
    encode_params_rotGroup = []


    for i in range(Nh): #TODO: SUPPORT different slots, Nh should be the value of slots
        encode_params_rotGroup.append(fivePows)
        fivePows = (fivePows * 5) % M

    # m_ksiPows stores the complex roots of unity
    for j in range(M):
        angle = 2.0 * M_PI * j / M
        encode_params_ksiPows.append(cmath.exp(1j * angle))
    encode_params_ksiPows.append(encode_params_ksiPows[0])

    # encode_params_ksiPows = np.array(encode_params_ksiPows, dtype=np.complex128).view(np.float64).tolist() #fixme: why it is correct previously
    encode_params_ksiPows = np.array(encode_params_ksiPows, dtype=np.complex128)
    encode_params_rotGroup = np.array(encode_params_rotGroup)

    # construction ends

    flag_i = False # align with openfhe

    slots = len(encode_params_rotGroup)

    if str(int(math.log2(slots))) not in cryptoContext.BsContext_map:
        error_msg = f"Precomputations for {slots} slots were not generated. Need to call EvalBootstrapSetup to proceed."
        raise ValueError(error_msg)

    level_budget = precom.paramsEnc.level_budget
    layers_collapse = precom.paramsEnc.layers_coll
    rem_collapse = precom.paramsEnc.layers_rem
    num_rotations = precom.paramsEnc.num_rotations
    b = precom.paramsEnc.baby_step
    g = precom.paramsEnc.giant_step
    num_rotations_rem = precom.paramsEnc.num_rotations_rem
    b_rem = precom.paramsEnc.baby_step_rem
    g_rem = precom.paramsEnc.giant_step_rem

    stop = -1
    flag_rem = 0

    if rem_collapse != 0:
        stop = 0
        flag_rem = 1

    result = [[] for _ in range(level_budget)]
    for i in range(level_budget):
        if flag_rem == 1 and i == 0:
            result[i] = [None] * num_rotations_rem
        else:
            result[i] = [None] * num_rotations

    # towers_to_drop = 0
    # if lRemain != 0:
    #     towers_to_drop = cryptoContext.L - lRemain - level_budget
    # for _ in range(towers_to_drop):
    #     element_params.pop_last_param()

    moduliQ_tmp = cryptoContext.moduliQ.clone()
    rootsQ_tmp = cryptoContext.power_of_roots.clone()
    if lRemain != 0:
        moduliQ_tmp = moduliQ_tmp[:lRemain+level_budget]
        rootsQ_tmp = rootsQ_tmp[:lRemain+level_budget]

    level0 = cryptoContext.L - lRemain - 1

    #combine the two tensors params_q and params_p
    params_q = moduliQ_tmp
    params_p = cryptoContext.primes[cryptoContext.L + 1:].clone()
    size_q = lRemain+level_budget
    size_p = cryptoContext.K
    primes = torch.cat((params_q, params_p), dim=0)

    roots_q = rootsQ_tmp
    roots_p = cryptoContext.power_of_roots.clone()[cryptoContext.L:]
    roots = torch.cat((roots_q, roots_p), dim=0)

    # primes = cryptoContext.primes,
    barret_ratio = cryptoContext.barret_ratio.clone()
    barret_k = cryptoContext.barret_k.clone()
    power_of_roots_shoup = cryptoContext.power_of_roots_shoup.clone()
    power_of_roots = cryptoContext.power_of_roots.clone()

    params_vector = [None] * (level_budget - stop)

    for s in range(level_budget - 1, stop - 1, -1):
        # Store a *copy* of moduli and roots at current level
        params_vector[s - stop] = {
            "primes": primes.clone(),
            "roots": roots.clone(),
            "barret_ratio": barret_ratio.clone(),
            "barret_k": barret_k.clone(),
            "power_of_roots_shoup": power_of_roots_shoup.clone(),
            "power_of_roots": power_of_roots.clone(),
        }

        # Remove the (size_q - 1)-th element
        index = size_q - 1
        primes = torch.cat((primes[:index], primes[index + 1:]), dim=0)
        roots = torch.cat((roots[:index], roots[index + 1:]), dim=0)
        barret_ratio = torch.cat((barret_ratio[:index], barret_ratio[index + 1:]), dim=0)
        barret_k = torch.cat((barret_k[:index], barret_k[index + 1:]), dim=0)
        power_of_roots_shoup = torch.cat((power_of_roots_shoup[:index], power_of_roots_shoup[index + 1:]), dim=0)
        power_of_roots = torch.cat((power_of_roots[:index], power_of_roots[index + 1:]), dim=0)
        size_q -= 1


    if slots == M // 4:
        coeff = coeff_encoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, flag_i) # the fft values

        for s in range(level_budget - 1, stop, -1):
            for i in range(b):
                for j in range(g):
                    if g * i + j != num_rotations:
                        rot = reduce_rotation(-g * i * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)), slots)
                        if flag_rem == 0 and s == stop + 1:
                            for k in range(slots):
                                coeff[s][g * i + j][k] *= scale

                        rotate_temp = rotate(coeff[s][g * i + j], rot)
                        cryptoContext.encode_params = copy.deepcopy(params_vector[s - stop])
                        result[s][g * i + j] = encode_bsMatrix(rotate_temp,f"coeff[{s}][{g * i + j}]_{rot}", level0 - s, len(rotate_temp), cryptoContext)

        if flag_rem:
            for i in range(b_rem):
                for j in range(g_rem):
                    if g_rem * i + j != num_rotations_rem:
                        rot = reduce_rotation(-g_rem * i, slots)
                        for k in range(slots):
                            coeff[stop][g_rem * i + j][k] *= scale

                        rotate_temp = rotate(coeff[stop][g_rem * i + j], rot)
                        cryptoContext.encode_params = params_vector[0]
                        result[stop][g_rem * i + j] = encode_bsMatrix(rotate_temp, f"coeff[{stop}][{g_rem * i + j}]_{rot}",
                                                               level0, len(rotate_temp), cryptoContext)

    else:
        coeff = coeff_encoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, False)
        coeffi = coeff_encoding_collapse(encode_params_ksiPows, encode_params_rotGroup, level_budget, True)

        for s in range(level_budget - 1, stop, -1):
            for i in range(b):
                for j in range(g):
                    if g * i + j != num_rotations:
                        rot = reduce_rotation(-g * i * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)), M // 4)
                        clear_temp = coeff[s][g * i + j] + coeffi[s][g * i + j]
                        if flag_rem == 0 and s == stop + 1:
                            for k in range(len(clear_temp)):
                                clear_temp[k] *= scale

                        rotate_temp = rotate(clear_temp, rot)

                        cryptoContext.encode_params = params_vector[s - stop]
                        result[s][g * i + j] = encode_bsMatrix(rotate_temp, f"clear_temp_{s}_{i}_{j}_{rot}",
                                                               level0 - s, len(rotate_temp), cryptoContext)

        if flag_rem:
            for i in range(b_rem):
                for j in range(g_rem):
                    if g_rem * i + j != num_rotations_rem:
                        rot = reduce_rotation(-g_rem * i, M // 4)
                        clear_temp = coeff[stop][g_rem * i + j] + coeffi[stop][g_rem * i + j]
                        for k in range(len(clear_temp)):
                            clear_temp[k] *= scale

                        rotate_temp = rotate(clear_temp, rot)
                        cryptoContext.encode_params = params_vector[0]
                        result[stop][g_rem * i + j] = encode_bsMatrix(rotate_temp, f"clear_temp_{i}_{j}_{rot}",
                                                               level0, len(rotate_temp), cryptoContext)
    print("temp")
    return result
