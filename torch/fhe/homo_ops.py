import warnings

from .ciphertext import Cipher
from . import functional as F
from .ciphertext import Plaintext as Plaintext
import math
import numpy as np
import torch
from .utils import check_meta_equal, check_cipher_len

BASE_NUM_LEVELS_TO_DROP = 1 #todo: to be removed?

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
        scf2 = cryptoContext.GetScalingFactorRealBig(rct2.cur_limbs + 2 - rct2.noise_deg)
        scf = cryptoContext.GetScalingFactorReal(rct1.cur_limbs)
        q1 = cryptoContext.GetModReduceFactor(rct1.cur_limbs - 1) if rct1.noise_deg == 2 else 1
        scaling_factor = scf2 / scf1 * q1 / scf
        if rct1.noise_deg == 2 and not (rct2.noise_deg == 1 and rct1.cur_limbs - rct2.cur_limbs == 1):
            rct1 = eval_mult_core(rct1, scaling_factor, cryptoContext)
            rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        elif rct1.noise_deg == 1:
            rct1 = eval_mult_core(rct1, scaling_factor, cryptoContext)
        else:
            raise ValueError
        rct1.try_drop_last_elements(rct1.cur_limbs - rct2.cur_limbs + rct2.noise_deg - 2)
        if rct2.noise_deg == 1:
            rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        rct1.scaling_factor = rct2.scaling_factor

        if swapped:
            rct1, rct2 = rct2.shallow_copy(), rct1.shallow_copy()
    else:
        if rct1.noise_deg < rct2.noise_deg:
            rct1 = eval_mult_core(rct1, 1.0, cryptoContext)
        elif rct2.noise_deg < rct1.noise_deg:
            rct2 = eval_mult_core(rct2, 1.0, cryptoContext)
    return rct1, rct2

# AdjustForAddOrSubInPlace in rns-leveledshe.cpp #todo: to check!!
def _adjust_for_add_or_sub(in0, in1, cryptoContext):
    rescaleTech = cryptoContext.rescaleTech
    if rescaleTech == "FIXEDMANUAL":
        # function `_adjust_levels` doesnt change memory, therefore its fine with a ciphertext morphed form plaintext
        in0, in1 = _adjust_levels(in0, in1, cryptoContext)
        if len(in0.cv) == len(in1.cv):
            return in0, in1

        scFactor = cryptoContext.GetScalingFactorReal(cryptoContext.L) #openfhe default value is 0, here transfer to the max value of #limb
        if scFactor == 0.0:
            raise ValueError("Unsupported scaling factor")

        if len(in0.cv)==1:
            ptxt = in0.cv[0]
            ptxtDepth = in0.noise_deg
            ctxtDepth = in1.noise_deg
            sizeQl = in1.cur_limbs
            moduli = cryptoContext.moduliQ[:sizeQl]
            ptxtIndex = 0
        elif len(in1.cv)==1:
            ptxt = in1.cv[0]
            ptxtDepth = in1.noise_deg
            ctxtDepth = in0.noise_deg
            sizeQl = in0.cur_limbs
            moduli = cryptoContext.moduliQ[:sizeQl]
            ptxtIndex = 1

        if len(in0.cv)==1 or len(in1.cv)==1: #todo: this branch is not tested
            # Bring to same depth if not already same
            if ptxtDepth < ctxtDepth:
                diffDepth = ctxtDepth - ptxtDepth
                intSF = int(scFactor + 0.5) # todo: to check if equivalent to openfhe
                crtSF = torch.tensor(sizeQl * [intSF], dtype=torch.uint64)
                crtPowSF = torch.clone(crtSF)
                for i in range(diffDepth):
                    crtPowSF = crt_mult(crtPowSF, crtSF, moduli)
                ptxt = F.cv_mul_scalar(
                    ptxt, crtPowSF.cuda(), cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, len(moduli)
                )   #fixme: crtPowSF should be a tensor for F.cv_mul_scalar? refactor crt_mult?

                if ptxtIndex == 0:
                    in0.cv[0] = ptxt # todo: check if correctly assigned
                    in0.noise_deg = ctxtDepth
                else:
                    in1.cv[0] = ptxt # todo: check if correctly assigned
                    in1.noise_deg = ctxtDepth
            elif ptxtDepth > ctxtDepth:
                raise ValueError("plaintext cannot be encoded at a larger depth than that of the ciphertext.")

    else:
        in0, in1 = adjust_levels_and_depth(in0, in1, cryptoContext)

    return in0, in1

#todo: only support in `FIXEDMANUAL` mode, or `adjust_levels_and_depth` function.
# should not be used directly in other rescale modes!!! except when openfhe directly used it
#todo: write homo_level_reduce

@check_meta_equal
def _cipher_add(in0, in1, cryptoContext):
    cv = [
        F.cv_add(cv0, cv1, cryptoContext.moduliQ_cuda, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg, in0.slots)

@check_meta_equal
def _cipher_sub(in0, in1, cryptoContext):
    cv = [
        F.cv_sub(cv0, cv1, cryptoContext.moduliQ_cuda, in0.cur_limbs)
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg, in0.slots)


@check_meta_equal
def _cipher_mul(in0, in1, cryptoContext):
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
    return Cipher([bx, ax, axax], in0.cur_limbs, in0.scaling_factor*scFactor, in0.noise_deg+in1.noise_deg, in0.slots)

@check_cipher_len
def _cipher_square(in0, cryptoContext):
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
    return Cipher([bx, ax, axax], in0.cur_limbs, in0.scaling_factor*scFactor, in0.noise_deg + in0.noise_deg, in0.slots)

@check_cipher_len
def _cipher_add_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ, in0.cur_limbs)
    cv = [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ_cuda, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs,in0.scaling_factor, in0.noise_deg, in0.slots)

@check_cipher_len
def _cipher_sub_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    cv = [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ_cuda, in0.cur_limbs),
        in0.cv[1],
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg, in0.slots)

#todo: used only in `homo_mul_scalar_int`, therefore the scaling factor and noise_deg remain unchanged
#todo: if used for `homo_mul_scalar_double`, the scaling factor and noise_deg should be changed
@check_cipher_len
def _cipher_mul_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_cuda, in0.cur_limbs)
    cv = [
        F.cv_mul_scalar(
            cv0, scalar_mod, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, in0.cur_limbs
        )
        for cv0 in in0.cv
    ]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg, in0.slots)

@check_cipher_len
def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ_cuda, in0.cur_limbs) for cv0 in in0.cv]
    return Cipher(cv, in0.cur_limbs, in0.scaling_factor, in0.noise_deg, in0.slots)

def homo_add(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_add(in0, in1, cryptoContext)

def homo_sub(in0, in1, cryptoContext):
    in0, in1 = _adjust_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)

def homo_mul(in0, in1, cryptoContext):
    # note: AdjustForMultInPlace in rns-leveledshe.cpp
    def adjust_for_mult(ct1: Cipher, ct2: Cipher, cryptoContext):
        if cryptoContext.rescaleTech == "FIXEDMANUAL":
            rct1,rct2 = _adjust_levels(ct1, ct2, cryptoContext)
        else:
            # inline `AdjustLevelsAndDepthToOneInPlace` in ckksrns-leveledshe.cpp as following
            rct1,rct2 = adjust_levels_and_depth(ct1, ct2, cryptoContext)
            if rct1.noise_deg==2:
                rct1 = homo_rescale(rct1, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                rct2 = homo_rescale(rct2, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        return rct1, rct2

    in0, in1 = adjust_for_mult(in0, in1, cryptoContext)
    if in0.slots != in1.slots:
        warnings.warn(f"slots unequal! in0.slots = {in0.slots}, in1.slots = {in1.slots}",
                      Warning)
    res = _cipher_mul(in0, in1, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx_cuda,
            cryptoContext.swk_ax_cuda,
            cryptoContext,
        ),
        cur_limbs=res.cur_limbs,
        scaling_factor=res.scaling_factor,
        noise_deg=res.noise_deg,
        slots=res.slots
    )
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)


def homo_square(in0, cryptoContext):
    in0 = in0.clone()
    if cryptoContext.rescaleTech != "FIXEDMANUAL" and in0.noise_deg != 1:
        in0 = homo_rescale(in0, 1, cryptoContext)
    res = _cipher_square(in0, cryptoContext)
    tmp = Cipher(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.swk_bx_cuda,
            cryptoContext.swk_ax_cuda,
            cryptoContext,
        ),
        cur_limbs=res.cur_limbs,
        scaling_factor=res.scaling_factor,
        noise_deg=res.noise_deg,
        slots=res.slots
    )
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)

def homo_rescale(ct, levels, cryptoContext):
    assert levels == 1 or levels == 0 and "Only support these two cases"
    if levels == 0: return ct.deep_copy()

    # ct1 = ct.clone()

    for l in range(levels):
        res0 = F.cv_drop_last_element_and_scale(ct.cv[0], ct.cur_limbs, l, cryptoContext)
        res1 = F.cv_drop_last_element_and_scale(ct.cv[1], ct.cur_limbs, l, cryptoContext)

    scFactor = ct.scaling_factor
    for l in range(levels):
        modReduceFactor = cryptoContext.GetModReduceFactor(ct.cur_limbs-1-l) # corresponding to openfhe: (sizeQl -1 -i), we need to use the value of input ct
        scFactor = scFactor/modReduceFactor

    return Cipher([res0, res1], ct.cur_limbs-levels, scFactor, ct.noise_deg-levels, ct.slots)

def cpp_round(_float, _len=0):
    i = int(_float)
    if isinstance(_float, float):
        return i if ((_float - i) < 0.5) else i + 1
    else:
        return round(_float, _len)

#todo: implement void EvalSubInPlace(Ciphertext<Element>& ciphertext, double constant) in cryptocontext.h?
def _get_element_for_eval_add_or_sub(ciphertext, constant, cryptoContext):
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

    crt_constant = torch.tensor(cur_limbs * [sc_constant], dtype=torch.uint64)

    # Scale back up by approxFactor within the CRT multiplications.
    if log_approx > 0:
        log_step = min(log_approx, LargeScalingFactorConstants.MAX_LOG_STEP.value)
        int_step = 2 ** log_step
        crt_approx = torch.tensor(cur_limbs * [int_step], dtype=torch.uint64)
        log_approx -= log_step

        while log_approx > 0:
            log_step = min(log_approx, LargeScalingFactorConstants.MAX_LOG_STEP.value)
            int_step = 2 ** log_step
            crt_sf = torch.tensor(cur_limbs * [int_step], dtype=torch.uint64)
            crt_approx = crt_mult(crt_approx, crt_sf, moduli)
            log_approx -= log_step

        crt_constant = crt_mult(crt_constant, crt_approx, moduli)

    # Handle FLEXIBLEAUTOEXT mode at level 0, we don't use the depth to calculate the scaling factor,
    # so we return the value before taking the depth into account.
    if cryptoContext.rescaleTech == 'FLEXIBLEAUTOEXT' and cur_limbs == sizeQ:
        return crt_constant

    # Final scaling factor adjustments
    int_sc_factor = int(sc_factor + 0.5)
    crt_sc_factor = torch.tensor(cur_limbs * [int_sc_factor], dtype=torch.uint64)

    for i in range(1, ciphertext.noise_deg):
        crt_constant = crt_mult(crt_constant, crt_sc_factor, moduli)

    return crt_constant

def homo_add_scalar_double(ct, cnst, cryptoContext):
    tmpr = _get_element_for_eval_add_or_sub(ct, math.fabs(cnst), cryptoContext) #tmpr should be a scalar vector, the following cipher function should be changed
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

    return Cipher(res, ct.cur_limbs, ct.scaling_factor, ct.noise_deg, ct.slots)

def homo_add_scalar(in0, scalar, cryptoContext):
    return _cipher_add_scalar(in0, scalar, cryptoContext)

#note: corresponds to MultByIntegerInPlace in openfhe, the datatype of scalar in openfhe is `uint64_t`
#fixme: do we accept scalar<0?
def homo_mul_scalar_int(in0, scalar, cryptoContext):
    abs_scalar = abs(scalar)
    res = _cipher_mul_scalar(in0, abs_scalar, cryptoContext)
    if scalar < 0:
        res = _cipher_neg(res, cryptoContext)
    return res


from enum import Enum

class LargeScalingFactorConstants(Enum):
    MAX_BITS_IN_WORD = 61
    MAX_LOG_STEP     = 60

# CRTMult in ckkspackedencoding.cpp
def crt_mult(a, b, mods):
    if len(a) != len(b) or len(a) != len(mods):
        raise ValueError("Input lists 'a', 'b', and 'mods' must have the same length.")

    result = torch.tensor([0] * len(a), dtype=torch.uint64)
    for i in range(len(mods)):
        result[i] = ((int(a[i]) * int(b[i])) % int(mods[i]))

    return result

# note: GetElementForEvalMult in ckksrns-leveledshe.cpp
def _get_element_for_eval_mult(cur_limbs, constant, cryptoContext):
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

    factors = torch.tensor([0] * num_towers, dtype=torch.uint64)
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
        crt_approx = torch.tensor(num_towers * [int_step], dtype=torch.uint64)
        log_approx -= log_step

        while log_approx > 0:
            log_step = log_approx if log_approx <= LargeScalingFactorConstants.MAX_LOG_STEP.value else LargeScalingFactorConstants.MAX_LOG_STEP.value
            int_step = 1 << log_step
            crt_sf = torch.tensor(num_towers * [int_step], dtype=torch.uint64)
            crt_approx = crt_mult(crt_approx, crt_sf, q_vec)
            log_approx -= log_step
        factors = crt_mult(factors, crt_approx, q_vec)

    return factors.cuda()

# note: EvalMultCoreInPlace in ckksrns-leveledshe.cpp
#todo: should merge this function with _cipher_mul_scalar? or redesign interface
def eval_mult_core(ciphertext, constant, cryptoContext):
    cur_limbs = ciphertext.cur_limbs
    factors = _get_element_for_eval_mult(cur_limbs, constant, cryptoContext)
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
    return Cipher(cv, ciphertext.cur_limbs, ciphertext.scaling_factor*scFactor, ciphertext.noise_deg+1, ciphertext.slots)

# note: EvalMultInPlace in ckksrns-leveledshe.cpp
def homo_mul_scalar_double(cipher, cnst, cryptoContext):
    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        if cipher.noise_deg == 2:
            cipher = homo_rescale(cipher, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return eval_mult_core(cipher, cnst, cryptoContext)

def homo_rotate(cipher, index, cryptoContext):
    auto_index = cryptoContext.find_auto_index(index)
    cur_limbs = cipher.cur_limbs
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    res = F.cv_keyswitch(cipher.cv[1], cur_limbs, swk[0], swk[1], cryptoContext)
    bxrot = F.cv_add(cipher.cv[0], res[0], cryptoContext.moduliQ_cuda, cur_limbs)

    # Apply the AutomorphismTransform to ax and bx
    cv0 = F.cv_automorphism_transform(bxrot, cur_limbs, auto_index, cryptoContext)
    cv1 = F.cv_automorphism_transform(res[1], cur_limbs, auto_index, cryptoContext)

    return Cipher([cv0, cv1], cur_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)

def homo_conjugate(cipher, cryptoContext):
    return homo_rotate(cipher, 2*cryptoContext.N-1, cryptoContext)

def homo_mul_pt(cipher, pt, cryptoContext):
    cur_limbs = cipher.cur_limbs
    if cipher.slots != pt.slots:
        warnings.warn(f"slots unequal! cipher.slots = {cipher.slots}, pt.slots = {pt.slots}",
                      Warning)
    if cipher.cur_limbs != pt.l:
        warnings.warn(f"limbs unequal! cipher.cur_limbs = {cipher.cur_limbs}, pt.l = {pt.l}, call adjust limbs function",
                      Warning)
    moduli = cryptoContext.moduliQ_cuda
    mu = cryptoContext.q_mu_cuda,
    cv0 = F.cv_mul(cipher.cv[0], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, cur_limbs)
    cv1 = F.cv_mul(cipher.cv[1], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, cur_limbs)
    return Cipher([cv0, cv1], cur_limbs, cipher.scaling_factor*pt.scaling_factor, cipher.noise_deg+pt.noise_deg, cipher.slots)

def homo_add_pt(cipher, pt, cryptoContext):
    ctmorphed = Cipher([pt.mx.reshape(-1, cryptoContext.N)], pt.l, pt.scaling_factor, pt.noise_deg, pt.slots) #MorphPlaintext in openfhe
    res0, res1 = _adjust_for_add_or_sub(cipher, ctmorphed, cryptoContext)
    res0.cv[0] = F.cv_add(res0.cv[0], res1.cv[0], cryptoContext.moduliQ_cuda, cipher.cur_limbs)
    return res0, res1