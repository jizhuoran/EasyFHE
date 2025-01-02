from .ciphertext import Cipher
from . import functional as F
from .ciphertext import Plaintext as Plaintext
import math
import numpy as np
import torch

BASE_NUM_LEVELS_TO_DROP = 1 #todo: to be removed?

# note: AdjustLevelsInPlace in rns-leveledshe.cpp
# mainly for "FIXEDMANUAL" case
def adjust_levels(ct1, ct2, cryptoContext):
    rct1 = Cipher([ct1.cv[0].clone(), ct1.cv[1].clone()], ct1.cur_limbs, ct1.scaling_factor, ct1.noise_deg)
    rct2 = Cipher([ct2.cv[0].clone(), ct2.cv[1].clone()], ct2.cur_limbs, ct2.scaling_factor, ct2.noise_deg)
    if rct1.cur_limbs > rct2.cur_limbs:
        rct1 = cipher_level_reduce(rct1, rct1.cur_limbs - rct2.cur_limbs)
    elif rct1.cur_limbs < rct2.cur_limbs:
        rct2 = cipher_level_reduce(rct2, rct2.cur_limbs - rct1.cur_limbs)
    return rct1, rct2

# note: AdjustLevelsAndDepthToOneInPlace in rns-leveledshe.cpp
def adjust_levels_and_depth(ct1, ct2, cryptoContext):
    rct1 = Cipher([ct1.cv[0].clone(), ct1.cv[1].clone()], ct1.cur_limbs, ct1.scaling_factor, ct1.noise_deg)
    rct2 = Cipher([ct2.cv[0].clone(), ct2.cv[1].clone()], ct2.cur_limbs, ct2.scaling_factor, ct2.noise_deg)
    L = cryptoContext.L
    c1_cur_limbs = rct1.cur_limbs
    c2_cur_limbs = rct2.cur_limbs
    c1lvl = L - c1_cur_limbs
    c2lvl = L - c2_cur_limbs
    c1depth = rct1.noise_deg
    c2depth = rct2.noise_deg
    sizeQl1 = c1_cur_limbs
    sizeQl2 = c2_cur_limbs

    if c1lvl < c2lvl:
        if c1depth == 2:
            if c2depth == 2:
                scf1 = rct1.scaling_factor
                scf2 = rct2.scaling_factor
                scf = cryptoContext.GetScalingFactorReal(cur_limbs =c1_cur_limbs)
                q1 = cryptoContext.GetModReduceFactor(sizeQl1 - 1)
                rct1 = eval_mult_core(rct1, scf2 / scf1 * q1 / scf, cryptoContext)
                rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                if (c1lvl+1<c2lvl):
                    rct1 =cipher_level_reduce(rct1, c2lvl - c1lvl - 1)
                rct1.scaling_factor = rct2.scaling_factor
            else:
                if c1lvl +1 ==c2lvl:
                    homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                else:
                    scf1 = rct1.scaling_factor
                    scf2 = cryptoContext.GetScalingFactorRealBig(cur_limbs = (L-(c2lvl-1)))
                    scf = cryptoContext.GetScalingFactorReal(cur_limbs = (L-c1lvl))
                    q1 = cryptoContext.GetModReduceFactor(sizeQl1 - 1)
                    rct1 = eval_mult_core(rct1, scf2 / scf1 * q1 / scf, cryptoContext)
                    rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                    if (c1lvl+2<c2lvl):
                        rct1 =cipher_level_reduce(rct1, c2lvl - c1lvl - 2)
                    rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                    rct1.scaling_factor = rct2.scaling_factor
        else:
            if c2depth==2:
                scf1 = rct1.scaling_factor
                scf2 = rct2.scaling_factor
                scf = cryptoContext.GetScalingFactorReal(cur_limbs =c1_cur_limbs)
                rct1 = eval_mult_core(rct1, scf2 / scf1 / scf, cryptoContext)
                rct1 = cipher_level_reduce(rct1, c2lvl - c1lvl)
                rct1.scaling_factor = scf2
            else:
                scf1 = rct1.scaling_factor
                scf2 = cryptoContext.GetScalingFactorRealBig(cur_limbs = (L-(c2lvl-1)))
                scf = cryptoContext.GetScalingFactorReal(cur_limbs = (L-c1lvl))
                rct1 = eval_mult_core(rct1, scf2 / scf1 / scf, cryptoContext)
                if (c1lvl+1<c2lvl):
                    rct1 = cipher_level_reduce(rct1, c2lvl - c1lvl - 1)
                rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                rct1.scaling_factor = rct2.scaling_factor
    elif c1lvl>c2lvl:
        if c2depth == 2:
            if c1depth == 2:
                scf2 = rct2.scaling_factor
                scf1 = rct1.scaling_factor
                scf = cryptoContext.GetScalingFactorReal(cur_limbs =c2_cur_limbs)
                q2 = cryptoContext.GetModReduceFactor(sizeQl2 - 1)
                rct2 = eval_mult_core(rct2, scf1 / scf2 * q2 / scf, cryptoContext)
                rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                if c2lvl+1 <c1lvl:
                    rct2 = cipher_level_reduce(rct2, c1lvl - c2lvl - 1)
                rct2.scaling_factor = rct1.scaling_factor
            else:
                if c2lvl+1 == c1lvl:
                    rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                else:
                    scf2 = rct2.scaling_factor
                    scf1 = cryptoContext.GetScalingFactorRealBig(cur_limbs = (L-(c1lvl-1)))
                    scf = cryptoContext.GetScalingFactorReal(cur_limbs = c2_cur_limbs)
                    q2 = cryptoContext.GetModReduceFactor(sizeQl2 - 1)
                    rct2 = eval_mult_core(rct2, scf1 / scf2 * q2 / scf, cryptoContext)
                    rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                    if c2lvl+2 < c1lvl:
                        rct2 = cipher_level_reduce(rct2, c1lvl - c2lvl - 2)
                    rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                    rct2.scaling_factor = rct1.scaling_factor
        else:
            if c1depth ==2:
                scf2 = rct2.scaling_factor
                scf1 = rct1.scaling_factor
                scf = cryptoContext.GetScalingFactorReal(cur_limbs =c2_cur_limbs)
                rct2 = eval_mult_core(rct2, scf1 / scf2 / scf, cryptoContext)
                rct2 = cipher_level_reduce(rct2, c1lvl - c2lvl)
                rct2.scaling_factor = scf1
            else:
                scf2 = rct2.scaling_factor
                scf1 = cryptoContext.GetScalingFactorRealBig(cur_limbs = (L-(c1lvl-1)))
                scf = cryptoContext.GetScalingFactorReal(cur_limbs = c2_cur_limbs)
                rct2 = eval_mult_core(rct2, scf1 / scf2 / scf, cryptoContext)
                if c2lvl+1 < c1lvl:
                    rct2 = cipher_level_reduce(rct2, c1lvl - c2lvl - 1)
                rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                rct2.scaling_factor = rct1.scaling_factor
    else:
        if c1depth < c2depth:
            rct1 = eval_mult_core(rct1, 1.0, cryptoContext)
        elif c2depth < c1depth:
            rct2 = eval_mult_core(rct2, 1.0, cryptoContext)
    return rct1, rct2


# note: AdjustForMultInPlace in rns-leveledshe.cpp
def adjust_for_mult(ct1: Cipher, ct2: Cipher, cryptoContext):
    rescaleTech = cryptoContext.rescaleTech

    if rescaleTech == "FIXEDMANUAL":
        rct1,rct2 = adjust_levels(ct1, ct2, cryptoContext)
    else:
        # inline `AdjustLevelsAndDepthToOneInPlace` in ckksrns-leveledshe.cpp as following
        rct1,rct2 = adjust_levels_and_depth(ct1, ct2, cryptoContext)
        if rct1.noise_deg==2:
            rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

    return rct1, rct2


# AdjustForAddOrSubInPlace in rns-leveledshe.cpp
def adjust_for_add_or_sub(in0, in1, cryptoContext):
    rescaleTech = cryptoContext.rescaleTech
    if rescaleTech == "FIXEDMANUAL":
        #fixme: function `adjust_levels` needs to support when input has class Plaintext!
        # or do some modifications here!
        rct1,rct2 = adjust_levels(in0, in1, cryptoContext)
        if isinstance(in0, Cipher) and isinstance(in1, Cipher):
            return rct1, rct2

        scFactor = cryptoContext.GetScalingFactorReal(cryptoContext.L) #openfhe default value is 0, here transfer to the max value of #limb
        if scFactor == 0.0:
            raise ValueError("Unsupported scaling factor")

        if isinstance(in0, Plaintext):
            ptxt = in0.mx
            ptxtDepth = in0.noise_deg
            ctxtDepth = in1.noise_deg
            sizeQl = in1.cur_limbs
            moduli = cryptoContext.moduliQ[:sizeQl]
            ptxtIndex = 0
        elif isinstance(in1, Plaintext):
            ptxt = in1.mx
            ptxtDepth = in1.noise_deg
            ctxtDepth = in0.noise_deg
            sizeQl = in0.cur_limbs
            moduli = cryptoContext.moduliQ[:sizeQl]
            ptxtIndex = 1

        if isinstance(in0, Plaintext) or isinstance(in1, Plaintext): #todo: this branch is not tested
            # Bring to same depth if not already same
            if ptxtDepth < ctxtDepth:
                diffDepth = ctxtDepth - ptxtDepth
                intSF = int(scFactor + 0.5) # todo: to check if equivalent to openfhe
                crtSF = np.full(sizeQl, intSF, dtype=np.uint64)
                crtPowSF = np.copy(crtSF)
                for i in range(diffDepth):
                    crtPowSF = crt_mult(crtPowSF, crtSF, moduli)

                F.cv_mul_scalar(
                    ptxt, crtPowSF, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, len(moduli)
                )   #fixme: crtPowSF should be a tensor for F.cv_mul_scalar? refactor crt_mult?

                if ptxtIndex == 0:
                    in0.mx = ptxt # todo: check if correctly assigned
                    in0.noise_deg = ctxtDepth
                else:
                    in1.mx = ptxt # todo: check if correctly assigned
                    in1.noise_deg = ctxtDepth
            elif ptxtDepth > ctxtDepth:
                raise ValueError("plaintext cannot be encoded at a larger depth than that of the ciphertext.")

    else:
        rct1,rct2 = adjust_levels_and_depth(in0, in1, cryptoContext)

    return rct1, rct2

# def cipher_rescale(ct, cryptoContext):  #todo: deprecated, to be removed, as well as inner functions
#     res0 = F.cv_rescale(ct.cv[0], cryptoContext, ct.cur_limbs)
#     res1 = F.cv_rescale(ct.cv[1], cryptoContext, ct.cur_limbs)
#     return Cipher([res0, res1], ct.cur_limbs - 1)

#todo: only support in `FIXEDMANUAL` mode, or `adjust_levels_and_depth` function.
# should not be used directly in other rescale modes!!! except when openfhe directly used it
#todo: write homo_level_reduce
def cipher_level_reduce(ct, levels):
    return Cipher(ct.cv, ct.cur_limbs - levels, ct.scaling_factor, ct.noise_deg)


def cipher_add(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_add(cv0, cv1, cryptoContext.moduliQ_cuda, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg)


def cipher_sub(in0, in1, cryptoContext):
    assert in0.cur_limbs == in1.cur_limbs
    cv = [
        F.cv_sub(cv0, cv1, cryptoContext.moduliQ_cuda, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg)


def cipher_mul(in0, in1, cryptoContext):
    assert len(in0.cv) == 2 and len(in1.cv) == 2
    assert in0.cur_limbs == in1.cur_limbs
    bx = F.cv_mul(
        in0.cv[0], in1.cv[0], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )
    ax = F.cv_add(
        F.cv_mul(
            in0.cv[0],
            in1.cv[1],
            cryptoContext.moduliQ_cuda,
            cryptoContext.q_mu_cuda,
            in0.cur_limbs,
        ),
        F.cv_mul(
            in0.cv[1],
            in1.cv[0],
            cryptoContext.moduliQ_cuda,
            cryptoContext.q_mu_cuda,
            in0.cur_limbs,
        ),
        cryptoContext.moduliQ_cuda,
        in0.cur_limbs,
    )
    axax = F.cv_mul(
        in0.cv[1], in1.cv[1], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )
    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return Cipher([bx, ax, axax], in0.cur_limbs, in0.scaling_factor*scFactor, in0.noise_deg+1)


def cipher_square(in0, cryptoContext):
    assert len(in0.cv) == 2
    bx = F.cv_mul(
        in0.cv[0], in0.cv[0], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )
    ax = F.cv_mul(
        in0.cv[0],
        in0.cv[1],
        cryptoContext.moduliQ_cuda,
        cryptoContext.q_mu_cuda,
        in0.cur_limbs,
    )
    ax = F.cv_add(ax, ax, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    axax = F.cv_mul(
        in0.cv[1], in0.cv[1], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
    )

    scFactor = cryptoContext.GetScalingFactorReal(in0.cur_limbs)
    return Cipher([bx, ax, axax], in0.cur_limbs, in0.scaling_factor*scFactor, in0.noise_deg + 1)


def cipher_add_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ, in0.cur_limbs)
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ_cuda, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs,in0.scaling_factor, in0.noise_deg)


def cipher_sub_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ_cuda, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg)

#todo: used only in `homo_mul_scalar_int`, therefore the scaling factor and noise_deg remain unchanged
#todo: if used for `homo_mul_scalar_double`, the scaling factor and noise_deg should be changed
def cipher_mul_scalar(in0, scalar, cryptoContext):
    assert len(in0.cv) == 2
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    cv = [
        F.cv_mul_scalar(
            cv0, scalar_mod, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
        )
        for cv0 in in0.cv
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg)


def cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ_cuda, in0.cur_limbs) for cv0 in in0.cv]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg)


def homo_add(in0, in1, cryptoContext):
    in0, in1 = adjust_for_add_or_sub(in0, in1, cryptoContext)
    return cipher_add(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    in0, in1 = adjust_for_add_or_sub(in0, in1, cryptoContext)
    return cipher_sub(in0, in1, cryptoContext)


def homo_mul(in0, in1, cryptoContext):
    in0, in1 = adjust_for_mult(in0, in1, cryptoContext)
    res = cipher_mul(in0, in1, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx_cuda,
            cryptoContext.swk_ax_cuda,
            cryptoContext,
        ),
        cur_limbs=in0.cur_limbs,
        scaling_factor=in0.scaling_factor*in1.scaling_factor,
        noise_deg=in0.noise_deg+in1.noise_deg,
    )
    res.cv = res.cv[:2]
    return cipher_add(res, tmp, cryptoContext)


def homo_square(in0, cryptoContext):
    in0 = in0.clone()
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg != 1:
        in0 = homo_rescale(in0, 1, cryptoContext)
    res = cipher_square(in0, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx_cuda,
            cryptoContext.swk_ax_cuda,
            cryptoContext,
        ),
        cur_limbs=in0.cur_limbs,
        scaling_factor=in0.scaling_factor * in0.scaling_factor,
        noise_deg=in0.noise_deg*2,
    )
    res.cv = res.cv[:2]
    return cipher_add(res, tmp, cryptoContext)

def homo_rescale(ct, levels, cryptoContext):
    if levels == 0: return Cipher(ct.cv, ct.cur_limbs, ct.scaling_factor, ct.noise_deg)

    curr_limbs = ct.cur_limbs
    for l in range(levels):
        res0 = F.cv_drop_last_element_and_scale(ct.cv[0], curr_limbs, l, cryptoContext)
        res1 = F.cv_drop_last_element_and_scale(ct.cv[1], curr_limbs, l, cryptoContext)

    curr_limbs -= levels
    noise_deg = ct.noise_deg-levels
    scFactor = ct.scaling_factor
    for l in range(levels):
        modReduceFactor = float(cryptoContext.GetModReduceFactor(ct.cur_limbs-1-l)) # corresponding to openfhe: (sizeQl -1 -i), we need to use the value of input ct
        scFactor = scFactor/modReduceFactor

    return Cipher([res0, res1], curr_limbs, scFactor, noise_deg)

def cpp_round(_float, _len=0):
    i = int(_float)
    if isinstance(_float, float):
        return i if ((_float - i) < 0.5) else i + 1
    else:
        return round(_float, _len)

#todo: implement void EvalSubInPlace(Ciphertext<Element>& ciphertext, double constant) in cryptocontext.h?
def get_element_for_eval_add_or_sub(ciphertext, constant, cryptoContext):
    cur_limbs = ciphertext.cur_limbs
    moduli = cryptoContext.moduliQ[:cur_limbs]
    sizeQ = cryptoContext.L
    # Scaling factor
    sc_factor = 0
    if cryptoContext.rescaleTech == 'FLEXIBLEAUTOEXT' and cur_limbs == sizeQ:
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

    crt_constant = np.full(cur_limbs, sc_constant, dtype=np.uint64)

    # Scale back up by approxFactor within the CRT multiplications.
    if log_approx > 0:
        log_step = min(log_approx, LargeScalingFactorConstants.MAX_LOG_STEP.value)
        int_step = 2 ** log_step
        crt_approx = np.full(cur_limbs, int_step, dtype=np.uint64)
        log_approx -= log_step

        while log_approx > 0:
            log_step = min(log_approx, LargeScalingFactorConstants.MAX_LOG_STEP.value)
            int_step = 2 ** log_step
            crt_sf = np.full(cur_limbs, int_step, dtype=np.uint64)
            crt_approx = crt_mult(crt_approx, crt_sf, moduli)
            log_approx -= log_step

        crt_constant = crt_mult(crt_constant, crt_approx, moduli)

    # Handle FLEXIBLEAUTOEXT mode at level 0, we don't use the depth to calculate the scaling factor,
    # so we return the value before taking the depth into account.
    if cryptoContext.rescaleTech == 'FLEXIBLEAUTOEXT' and cur_limbs == sizeQ:
        return crt_constant

    # Final scaling factor adjustments
    int_sc_factor = int(sc_factor + 0.5)
    crt_sc_factor = np.full(cur_limbs, int_sc_factor, dtype=np.uint64)

    for i in range(1, ciphertext.noise_deg):
        crt_constant = crt_mult(crt_constant, crt_sc_factor, moduli)

    return crt_constant

def homo_add_scalar_double(ct, cnst, cryptoContext):
    tmpr = get_element_for_eval_add_or_sub(ct, math.fabs(cnst), cryptoContext) #tmpr should be a scalar vector, the following cipher function should be changed
    tmpr_tensor = torch.from_numpy(
        np.array(
            [int(int(tmpr[l]) % int(cryptoContext.moduliQ[l])) for l in range(ct.cur_limbs)],
            dtype=np.uint64,
        )
    ).cuda()
    if cnst < 0:
        res = [
            F.cv_sub_scalar(ct.cv[0], tmpr_tensor, cryptoContext.moduliQ_cuda, ct.cur_limbs),
            ct.cv[1],
        ]
    else:
        res = [
            F.cv_add_scalar(ct.cv[0], tmpr_tensor, cryptoContext.moduliQ_cuda, ct.cur_limbs),
            ct.cv[1],
        ]

    return Cipher(res, ct.cur_limbs, ct.scaling_factor, ct.noise_deg)

    # deprecated version
# def homo_add_scalar_double(ct, cnst, cryptoContext):
    # tmpr = cpp_round(abs(cnst) * (2 ** cryptoContext.logqi))
    # if cnst < 0:
    #     res = cipher_sub_scalar(ct, tmpr1[0], cryptoContext).cv
    # else:
    #     res = cipher_add_scalar(ct, tmpr1[0], cryptoContext).cv
    # return Cipher(res, ct.cur_limbs)

#note: corresponds to MultByIntegerInPlace in openfhe, the datatype of scalar in openfhe is `uint64_t`
#fixme: should call `abs` before `cipher_mul_scalar` first, and then `cipher_mul_scalar`; or prohibit scalar<0
def homo_mul_scalar_int(in0, scalar, cryptoContext):
    res = cipher_mul_scalar(in0, scalar, cryptoContext)
    if scalar < 0:
        res = cipher_neg(res, cryptoContext)
    return res


from enum import Enum

class LargeScalingFactorConstants(Enum):
    MAX_BITS_IN_WORD = 61
    MAX_LOG_STEP     = 60

# CRTMult in ckkspackedencoding.cpp
def crt_mult(a, b, mods):
    if len(a) != len(b) or len(a) != len(mods):
        raise ValueError("Input lists 'a', 'b', and 'mods' must have the same length.")

    #fixme: should be a tensor?
    result = np.zeros(len(a), dtype=np.uint64)
    for i in range(len(mods)):
        result[i] = ((int(a[i]) * int(b[i])) % int(mods[i]))

    return result

# note: GetElementForEvalMult in ckksrns-leveledshe.cpp
def get_element_for_eval_mult(factors, cur_limbs, constant, cryptoContext):
    num_towers = cur_limbs
    q_vec = cryptoContext.moduliQ  # Assuming qVec is a numpy array
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

    factors = np.zeros(num_towers, dtype=np.uint64) #todo: allocate inside or outside? or remove outside allocation
    if large_abs >= bound:
        for i in range(num_towers):
            reduced = large % q_vec[i]
            factors[i] = reduced + q_vec[i] if reduced < 0 else reduced
    else:
        sc_constant = int(large)
        for i in range(num_towers):
            reduced = sc_constant % int(q_vec[i])
            factors[i] = reduced + q_vec[i] if reduced < 0 else reduced

    # Scale back up by approxFactor within the CRT multiplications.
    if log_approx > 0:
        log_step = log_approx if log_approx <= LargeScalingFactorConstants.MAX_LOG_STEP.value else LargeScalingFactorConstants.MAX_LOG_STEP.value
        int_step = 1 << log_step
        crt_approx = np.full(num_towers, int_step, dtype=np.uint64)
        log_approx -= log_step

        while log_approx > 0:
            log_step = log_approx if log_approx <= LargeScalingFactorConstants.MAX_LOG_STEP.value else LargeScalingFactorConstants.MAX_LOG_STEP.value
            int_step = 1 << log_step
            crt_sf = np.full(num_towers, int_step, dtype=np.uint64)
            crt_approx = crt_mult(crt_approx, crt_sf, q_vec)
            log_approx -= log_step
        factors = crt_mult(factors, crt_approx, q_vec)

    return factors

# note: EvalMultCoreInPlace in ckksrns-leveledshe.cpp
#todo: should merge this function with cipher_mul_scalar? or redesign interface
def eval_mult_core(ciphertext, constant, cryptoContext):
    cur_limbs = ciphertext.cur_limbs
    factors = np.zeros(cur_limbs, dtype=np.uint64)
    factors = get_element_for_eval_mult(factors, cur_limbs, constant, cryptoContext)
    factors = torch.tensor(factors, dtype=torch.uint64, device="cuda")
    cv = [
        F.cv_mul_scalar(
            cv_i,
            factors,
            cryptoContext.moduliQ_cuda,
            cryptoContext.q_mu_cuda,
            ciphertext.cur_limbs,
        )
        for cv_i in ciphertext.cv
    ]

    scFactor = cryptoContext.GetScalingFactorReal(cur_limbs)
    return Cipher(cv, ciphertext.cur_limbs, ciphertext.scaling_factor*scFactor, ciphertext.noise_deg+1)

# note: EvalMultInPlace in ckksrns-leveledshe.cpp
def homo_mul_scalar_double(cipher, cnst, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        if cipher.noise_deg == 2:
            cipher = homo_rescale(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return eval_mult_core(cipher, cnst, cryptoContext)

# def homo_mul_scalar_double(in0, scalar, cryptoContext):
#     tmpr = cpp_round(abs(scalar) * (2 ** cryptoContext.logqi))
#     res = cipher_mul_scalar(in0, tmpr, cryptoContext)
#     if scalar < 0:
#         res = cipher_neg(res, cryptoContext)
#     return Cipher(res.cv, in0.cur_limbs)

def homo_rotate(cipher, auto_index, ctx):
    cur_limbs = cipher.cur_limbs
    swk = ctx.left_rot_key_map[str(auto_index)]
    res = F.cv_keyswitch(cipher.cv[1], cur_limbs, swk[0], swk[1], ctx)
    bxrot = F.cv_add(cipher.cv[0], res[0], ctx.moduliQ_cuda, cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    cv0 = F.cv_automorphism_transform(bxrot, cur_limbs, auto_index, ctx)
    cv1 = F.cv_automorphism_transform(res[1], cur_limbs, auto_index, ctx)

    return Cipher([cv0, cv1], cur_limbs, cipher.scaling_factor, cipher.noise_deg)

def homo_conjugate(cipher, auto_index, ctx):
    return homo_rotate(cipher, auto_index, ctx)