import time

from .ciphertext import Cipher
from .ciphertext import Plaintext
from . import functional as F
from . import hybrid_keyswitch
import math
import functools
import torch
from .utils import (
    check_meta_equal,
    check_cipher_len,
    call_counter,
    profile_python_function,
)
from .compiler.compiler import frontend
import warnings


BASE_NUM_LEVELS_TO_DROP = 1  # todo: to be removed?


# drop last elem is a inplace operation now
@frontend
def drop_last_elements(ct, num_levels, cryptoContext, inplace=False):
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
        ct1 = drop_last_elements(
            ct1, ct1.cur_limbs - ct2.cur_limbs, cryptoContext, inplace=True, printInfo=False
        )
    elif ct1.cur_limbs < ct2.cur_limbs:
        ct2 = drop_last_elements(
            ct2, ct2.cur_limbs - ct1.cur_limbs, cryptoContext, inplace=True, printInfo=False
        )
    return ct1, ct2


def flex_adjust_to(
    cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
):
    # if cipher.cur_limbs != target_limbs:
    #     print("NODE0 = flex_adjust_to(NODE0) # limb {}".format(cipher.cur_limbs))
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
            cipher = homo_rescale_internal(
                cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
            )
            if cipher.cur_limbs > target_limbs:
                cipher = drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs,
                    cryptoContext,
                    inplace=True,
                    printInfo=False,
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
                cipher = drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs - 1,
                    cryptoContext,
                    inplace=True,
                    printInfo=False,
                )
            cipher = homo_rescale_internal(
                cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
            )
            cipher.scaling_factor = target_scaling_factor
            # so the output has noise_deg 1, and the same cur_limb
        elif cipher.noise_deg == 2 and target_noise_deg == 1:
            # if ct1 has degree 2, and it is just 1 more limb, do a rescale (seems this is the case the smae as fix?)
            if cipher.cur_limbs == target_limbs + 1:
                homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
                )
            else:
                # otherwise, mul the higher with scale factor, rescale, drop, rescale.
                # the last rescale is to make sure both has degree 1
                scf1 = cipher.scaling_factor
                scf2 = cryptoContext.GetScalingFactorRealBig(
                    cryptoContext.L - (cryptoContext.L - target_limbs - 1)
                )
                scf = cryptoContext.GetScalingFactorReal(cipher.cur_limbs)
                q1 = cryptoContext.GetModReduceFactor(cipher.cur_limbs - 1)
                cipher = _eval_mult_core(cipher, scf2 / scf1 * q1 / scf, cryptoContext)
                cipher = homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
                )
                if cipher.cur_limbs > target_limbs + 1:
                    cipher = drop_last_elements(
                        cipher,
                        cipher.cur_limbs - target_limbs - 1,
                        cryptoContext,
                        inplace=True,
                        printInfo=False,
                    )
                cipher = homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
                )
                cipher.scaling_factor = target_scaling_factor
        elif cipher.noise_deg == 1 and target_noise_deg == 2:
            # if the higher has a lower degree, mul it with the factor, and then drop
            # so the output has noise_deg 2, and same cur_limb
            scf1 = cipher.scaling_factor
            scf2 = target_scaling_factor
            scf = cryptoContext.GetScalingFactorReal(cipher.cur_limbs)
            cipher = _eval_mult_core(cipher, scf2 / scf1 / scf, cryptoContext)
            cipher = drop_last_elements(
                cipher, cipher.cur_limbs - target_limbs, cryptoContext, inplace=True, printInfo=False
            )
            cipher.scaling_factor = scf2
        else:
            print("noise_deg", cipher.noise_deg, target_noise_deg)
            raise ValueError

    # print("cipher", cipher.cur_limbs, cipher.noise_deg)
    return cipher


def fixed_adjust_to(
    cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
):
    # if cipher.cur_limbs != target_limbs:
    #     print("NODE0 = flex_adjust_to(NODE0) # limb {}".format(cipher.cur_limbs))

    assert cipher.cur_limbs >= target_limbs
    if cipher.cur_limbs == target_limbs:
        if cipher.noise_deg < target_noise_deg:
            return _eval_mult_core(cipher, 1.0, cryptoContext)
        else:
            return cipher.shallow_copy()
    else:  # cur_limbs > target_limbs
        if cipher.noise_deg == 2 and target_noise_deg == 2:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            cipher = homo_rescale_internal(
                cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
            )
            if cipher.cur_limbs > target_limbs:
                cipher = drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs,
                    cryptoContext,
                    inplace=False,
                    printInfo=False,
                )
        elif cipher.noise_deg == 1 and target_noise_deg == 1:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            if cipher.cur_limbs > target_limbs + 1:
                cipher = drop_last_elements(
                    cipher,
                    cipher.cur_limbs - target_limbs - 1,
                    cryptoContext,
                    inplace=True,
                    printInfo=False,
                )
            cipher = homo_rescale_internal(
                cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
            )
        elif cipher.noise_deg == 2 and target_noise_deg == 1:
            if cipher.cur_limbs == target_limbs + 1:
                homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
                )
            else:
                cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
                cipher = homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
                )
                if cipher.cur_limbs > target_limbs + 1:
                    cipher = drop_last_elements(
                        cipher,
                        cipher.cur_limbs - target_limbs - 1,
                        cryptoContext,
                        inplace=True,
                        printInfo=False,
                    )
                cipher = homo_rescale_internal(
                    cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
                )
        elif cipher.noise_deg == 1 and target_noise_deg == 2:
            cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
            cipher = drop_last_elements(
                cipher, cipher.cur_limbs - target_limbs, cryptoContext, inplace=True, printInfo=False
            )
        else:
            print("noise_deg", cipher.noise_deg, target_noise_deg)
            raise ValueError
    return cipher


# def flex_adjust_to(
#     cipher, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
# ):
#     assert cipher.cur_limbs >= target_limbs
#     while cipher.cur_limbs > target_limbs:
#         if cipher.noise_deg == 1:
#             cipher = _eval_mult_core(cipher, 1.0, cryptoContext)
#         cipher = homo_rescale_internal(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False)
#     if cipher.noise_deg < target_noise_deg:
#         res = _eval_mult_core(cipher, 1.0, cryptoContext)
#     else:
#         res = cipher
#     assert res.cur_limbs == target_limbs and res.noise_deg == target_noise_deg
#     return res


@frontend
def adjust_levels_and_depth(ct1, ct2, cryptoContext):
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
    if cryptoContext.rescaleTech == "FLEXIBLEAUTO":
        return flex_adjust_to(
            ct1, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
        ), flex_adjust_to(
            ct2, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
        )
    elif cryptoContext.rescaleTech == "FIXEDAUTO":
        return fixed_adjust_to(
            ct1, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
        ), fixed_adjust_to(
            ct2, target_limbs, target_noise_deg, target_scaling_factor, cryptoContext
        )
    else:
        raise ValueError


# AdjustForAddOrSubInPlace in rns-leveledshe.cpp
def _adjust_for_add_or_sub(in0, in1, cryptoContext):
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        assert in0.noise_deg == in1.noise_deg
        return _adjust_levels(in0, in1, cryptoContext)
    else:
        return adjust_levels_and_depth(in0, in1, cryptoContext, printInfo=False)


def _adjust_for_mult(ct1: Cipher, ct2: Cipher, cryptoContext):
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        rct1, rct2 = _adjust_levels(ct1, ct2, cryptoContext)
    else:
        # inline `AdjustLevelsAndDepthToOneInPlace` in ckksrns-leveledshe.cpp as following
        rct1, rct2 = adjust_levels_and_depth(ct1, ct2, cryptoContext, printInfo=False)
        if rct1.noise_deg == 2:
            rct1 = homo_rescale_internal(
                rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
            )
        if rct2.noise_deg == 2: #if rct1's noise_deg is 1, so is rct2
            rct2 = homo_rescale_internal(
                rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
            )
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
        crt_constant = crt_mult(
            crt_constant, crt_sc_factor, cryptoContext.moduliQ_scalar
        )

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
            cryptoContext.QplusP_map[in0.cur_limbs],
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
    return in0.cipher_like(
        [bx, ax, axax],
        scaling_factor=in0.scaling_factor * scFactor,
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
    return in0.cipher_like(
        [bx, ax, axax],
        scaling_factor=in0.scaling_factor * scFactor,
        noise_deg=in0.noise_deg + in0.noise_deg,
    )


@check_cipher_len
def _cipher_add_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(
        scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs
    )
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


@check_cipher_len
def _cipher_sub_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(
        scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs
    )
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ]
    return in0.cipher_like(cv)


# todo: used only in `homo_mul_scalar_int`, therefore the scaling factor and noise_deg remain unchanged
# todo: if used for `homo_mul_scalar_double`, the scaling factor and noise_deg should be changed
# @check_cipher_len #fixme: comment it to support call from homo_mul_pt
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
    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return in0.cipher_like(
        cv, scaling_factor=in0.scaling_factor * scFactor, noise_deg=in0.noise_deg + 1
    )


@check_cipher_len
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
    return in0.cipher_like(
        cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg
    )


@check_cipher_len
def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return in0.cipher_like(
        cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg
    )


# @check_cipher_len
# todo: input len of in0.cv could be 1
@frontend
def _cipher_automorphism(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    limbs = in0.cur_limbs if in0.is_ext == False else in0.cur_limbs + cryptoContext.K
    cv = [
        F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext)
        for cv in in0.cv
    ]
    return in0.cipher_like(cv)


@call_counter
@frontend
def homo_add(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_add_ext(in0, in1, cryptoContext)
    else:
        return _cipher_add(in0, in1, cryptoContext)


@call_counter
@frontend
def homo_sub(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)


@call_counter
@frontend
def homo_rescale(
    ct, levels, cryptoContext
):  # todo: add force_rescale flag in user api for other rescaleTech?
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        return homo_rescale_internal(ct, levels, cryptoContext, printInfo=False)
    else:
        return ct.deep_copy()


@frontend
@call_counter
def homo_rescale_internal(ct, levels, cryptoContext):
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

    return ct.cipher_like(
        res_cv,
        cur_limbs=ct.cur_limbs - levels,
        scaling_factor=scFactor,
        noise_deg=ct.noise_deg - levels,
    )


@call_counter
@frontend
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


@call_counter
@frontend
def homo_square(in0, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg != 1:
        in0 = homo_rescale_internal(in0, 1, cryptoContext, printInfo=False)
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


@frontend
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


# note: corresponds to MultByIntegerInPlace in openfhe, the datatype of scalar in openfhe is `uint64_t`
# fixme: do we accept scalar<0?
@frontend
def homo_mul_scalar_int(in0, scalar, cryptoContext):
    res = _cipher_mul_scalar_int(in0, abs(scalar), cryptoContext)
    if scalar < 0:
        res = _cipher_neg(res, cryptoContext)
    return res


# note: EvalMultInPlace in ckksrns-leveledshe.cpp
@call_counter
@frontend
def homo_mul_scalar_double(in0, cnst, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg == 2:
        in0 = homo_rescale_internal(
            in0, BASE_NUM_LEVELS_TO_DROP, cryptoContext, printInfo=False
        )
    return _eval_mult_core(in0, cnst, cryptoContext)


@frontend
def homo_rotate(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.get_rotation_key(norm_index)
    res = in0.cipher_like(
        F.cv_keyswitch(in0.cv[1], in0.cur_limbs, swk[0], swk[1], cryptoContext)
    )

    res.cv[0] = F.cv_add(in0.cv[0], res.cv[0], cryptoContext.moduliQ, in0.cur_limbs)

    res = _cipher_automorphism(res, index, cryptoContext, printInfo=False)

    return res


@frontend
def eval_fast_rotate(digits, cipher, index, need_KS_add, need_moddown, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    result = hybrid_keyswitch.mult_rot_key_and_sum_ext(
        digits, index, cryptoContext, printInfo=False
    )

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

    result = _cipher_automorphism(result, index, cryptoContext, printInfo=False)

    return result


def homo_conjugate(in0, cryptoContext):
    return homo_rotate(in0, 2 * cryptoContext.N - 1, cryptoContext)


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


@frontend
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
        cv0 = F.cv_mul(
            cipher.cv[0], plaintext.cv, moduli, mu, cipher.cur_limbs + cryptoContext.K
        )
        cv1 = F.cv_mul(
            cipher.cv[1], plaintext.cv, moduli, mu, cipher.cur_limbs + cryptoContext.K
        )
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


@frontend
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


def bit_reverse(vals):
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

def bitreverse(index: int, n_power: int) -> int:
    res_1 = 0
    for _ in range(n_power):
        res_1 <<= 1
        res_1 |= (index & 1)
        index >>= 1
    return res_1


def fft_special_inv(vals, M, rotGroup, ksiPows):
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

    vals = bit_reverse(vals)

    for i in range(vals_size):
        vals[i] /= vals_size
    return vals

def ptx_encode_cuda(
    x,
    slots,
    type_flag,
    scaling_factor,
    cur_limbs,
    noise_scale_deg,
    use_gpu_fft,
    cryptocontext,
):
    inverse = x
    pt_encode = []

    if slots < len(inverse):
        raise ValueError(
            f"The number of slots [{slots}] is less than the size of data [{len(inverse)}]"
        )

    if not use_gpu_fft:
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
        arr = cryptocontext.encode_params_ksiPows.cpu().numpy()
        complex_arr = arr[0::2] + arr[1::2]*1j
        inverse_complex = fft_special_inv(
            inverse_complex,
            cryptocontext.M,
            cryptocontext.encode_params_rotGroup.cpu().numpy(),
            complex_arr,
        )
        inverse_array = np.array(inverse_complex, dtype=np.complex128).view(np.float64).tolist()
        inverse_internal = torch.tensor(inverse_array, dtype=torch.double, device="cuda")
        inverse = torch.tensor(inverse, device="cuda")
    else:
        inverse_internal = cryptocontext.encode_inverse

    pt_encode = torch.encode(
        inverse=inverse,
        inverse_internal= inverse_internal,
        temp=cryptocontext.encode_temp,
        primes=cryptocontext.primes,
        precompute_rotgroups=cryptocontext.encode_params_rotGroup,
        precompute_ksipows=cryptocontext.encode_params_ksiPows,
        M=cryptocontext.M,
        N=cryptocontext.N,
        cur_limbs=cur_limbs,
        slots=slots,
        noise_scale_deg=noise_scale_deg,
        scaling_factor=scaling_factor,
        power_of_roots_shoup=cryptocontext.power_of_roots_shoup,
        power_of_roots=cryptocontext.power_of_roots,
        use_fft=use_gpu_fft,
    )

    return pt_encode


def encode(
    x, scale_deg=None, level=None, slots=None, use_gpu_fft=True, cryptoContext=None,
):
    if cryptoContext is None:
        raise ValueError("Error: cryptoContext is not set.")
    if not (
        (scale_deg is None and level is None and slots is None)
        or (scale_deg is not None and level is not None and slots is not None)
    ):
        raise ValueError(
            "Error: check if scale_deg, level, and slots are set correctly."
        )

    slots = (
        cryptoContext.Nh if slots is None else slots
    )  # note: default slots is N/2, which is Nh
    scale_deg = 1 if scale_deg is None else scale_deg
    cur_limb = cryptoContext.L if level is None else cryptoContext.L - level
    if cryptoContext.rescaleTech == "FLEXIBLEAUTOEXT":
        scFact = cryptoContext.GetScalingFactorRealBig(cur_limb)
        # In FLEXIBLEAUTOEXT mode at level 0, we don't use the noiseScaleDeg
        # in our encoding function, so we set it to 1 to make sure it
        # has no effect on the encoding.
        scale_deg = 1
    else:
        scFact = cryptoContext.GetScalingFactorReal(cur_limb)

    encoded_vector_dcrt_elements_cuda = ptx_encode_cuda(
        x, slots, "IsDCRTPoly", scFact, cur_limb, scale_deg, use_gpu_fft, cryptoContext
    )

    mv = [encoded_vector_dcrt_elements_cuda]
    return Plaintext(mv, mv[0].shape[0], scFact, scale_deg, slots, False)