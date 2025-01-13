import time, os
import warnings
from .ciphertext import Cipher
from .ciphertext import Plaintext as Plaintext
from .client.gen_context import gen_contexts
from .context import *
from .bs_context import *
from . import functional as F
from . import homo_ops
from . import approx as approx
from . import hoisting_keyswitch
from . import utils
import numpy as np

import torch.profiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler

Tensor = torch.Tensor
NORMAL_CIPHER_SIZE = 2
BASE_NUM_LEVELS_TO_DROP = 1
R_UNIFORM = 6  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
R_SPARSE = 3  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.

# @profile_python_function
def adjust_ciphertext(ciphertext, correction, L0, cryptoContext):
    rescale_tech = cryptoContext.rescaleTech

    if rescale_tech == "FLEXIBLEAUTO" or rescale_tech == "FLEXIBLEAUTOEXT":
        lvl = 0 if rescale_tech == "FLEXIBLEAUTO" else 1
        if cryptoContext.L != L0:
            # Print error message and raise an exception to stop the program
            print("cryptoContext.L != L0")
            raise Exception("Error: cryptoContext.L != L0")
        target_sf = cryptoContext.GetScalingFactorReal(cur_limbs = (L0 - lvl))
        source_sf = ciphertext.scaling_factor
        num_towers = len(ciphertext.cv)
        mod_to_drop = float(cryptoContext.moduliQ[num_towers - 1])
        # in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        # a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        # So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        # for NATIVEINT = 128.
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        adjustment_factor = (target_sf / source_sf) * (mod_to_drop / source_sf) * math.pow(2, -correction) # if NATIVEINT != 128
        ciphertext = homo_ops.homo_mul_scalar_double(ciphertext, adjustment_factor, cryptoContext)
        ciphertext = homo_ops.homo_rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ciphertext.scaling_factor = target_sf

    else:
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        cnst = math.pow(2, -correction)
        ciphertext = homo_ops.homo_mul_scalar_double(ciphertext, cnst, cryptoContext)
        ciphertext = homo_ops.homo_rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return ciphertext


# @profile_python_function
def apply_double_angle_iterations(ciphertext, cryptoContext):
    if cryptoContext.secretKeyDist == "UNIFORM_TERNARY":
        r = R_UNIFORM
    elif cryptoContext.secretKeyDist == "SPARSE_TERNARY":
        r = R_SPARSE
    else:
        raise ValueError("set secretKeyDist first!")

    for j in range(1, r + 1):
        ciphertext = homo_ops.homo_square(ciphertext, cryptoContext)
        ciphertext = homo_ops.homo_add(ciphertext, ciphertext, cryptoContext)
        scalar = -1.0 / math.pow((2.0 * math.pi), math.pow(2.0, j - r))
        ciphertext = homo_ops.homo_add_scalar_double(ciphertext, scalar, cryptoContext)
        ciphertext = homo_ops.homo_rescale(ciphertext, 1,
                                           cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else ciphertext
    return ciphertext


def merged_function(A, ctxt, cryptoContext, flag_rem, rot_in, rot_out, config):
    def key_switch_ext(cipher, add_first, cryptoContext):
        curr_limbs = cipher.cur_limbs

        cv0 = torch.zeros(((curr_limbs + cryptoContext.K) << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
        cv1 = torch.zeros(((curr_limbs + cryptoContext.K) << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
        if add_first:
            cv0[:curr_limbs, :] = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                                  cryptoContext.q_mu_cuda, curr_limbs)

        cv1[:curr_limbs, :] = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                              cryptoContext.q_mu_cuda, curr_limbs)
        return Cipher([cv0, cv1], curr_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)
    #todo: it is ct*pt in extent form, refactor?
    def eval_mult_ext(cipher, pt, cryptoContext):
        cur_limbs = cipher.cur_limbs
        if cipher.slots != pt.slots:
            warnings.warn(f"slots unequal! cipher.slots = {cipher.slots}, pt.slots = {pt.slots}",
                          Warning)
        moduli = cryptoContext.BsContext.QplusP_map[cur_limbs]
        mu = cryptoContext.BsContext.QmuplusPmu_map[cur_limbs]
        limbsExt = cur_limbs + cryptoContext.K
        cv0 = F.cv_mul(cipher.cv[0], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, limbsExt)
        cv1 = F.cv_mul(cipher.cv[1], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, limbsExt)
        return Cipher([cv0, cv1], cur_limbs, cipher.scaling_factor*pt.scaling_factor, cipher.noise_deg+pt.noise_deg, cipher.slots)

    def eval_add_ext(cipher0, cipher1, cryptoContext):
        assert cipher0.cur_limbs == cipher1.cur_limbs
        if cipher0.slots != cipher1.slots:
            warnings.warn(f"slots unequal! cipher0.slots = {cipher0.slots}, cipher1.slots = {cipher1.slots}",
                          Warning)
        limbsExt = cipher0.cur_limbs + cryptoContext.K
        moduli = cryptoContext.BsContext.QplusP_map[cipher0.cur_limbs]
        cv = [
            F.cv_add(cv0, cv1, moduli, limbsExt)
            for cv0, cv1 in zip(cipher0.cv, cipher1.cv)
        ]
        return Cipher(cv, cipher0.cur_limbs, cipher0.scaling_factor, cipher0.noise_deg, cipher0.slots)

    # @profile_python_function
    def cv_add_ext(in0, in1, cur_limbs, cryptoContext):
        moduli = cryptoContext.BsContext.QplusP_map[cur_limbs]
        res = F.cv_add(in0, in1, moduli, in0.shape[0])
        return res

    special_limbs = cryptoContext.K
    logN = cryptoContext.logN

    # Set up configuration
    loop_direction = config["loop_direction"]
    cipher_mod_levels = config["cipher_mod_levels"]
    eval_fast_rotation_reshape = config["eval_fast_rotation_reshape"]
    key_switch_ext_size = config["key_switch_ext_size"]
    params = config["params"]
    start = config["start"]
    stop = config["stop"]

    level_budget = params.level_budget
    num_rotations = params.num_rotations
    b = params.baby_step
    g = params.giant_step
    num_rotations_rem = params.num_rotations_rem
    g_rem = params.giant_step_rem
    b_rem = params.baby_step_rem

    result = Cipher([ctxt.cv[0].clone(), ctxt.cv[1].clone()], ctxt.cur_limbs, ctxt.scaling_factor, ctxt.noise_deg, ctxt.slots)

    # Determine loop range based on direction
    if loop_direction == "forward":
        loop_range = range(start, stop)
    else:
        loop_range = range(start, stop, -1)
    
    for s in loop_range:
        if (loop_direction == "forward" and s != 0) or (loop_direction == "backward" and s != level_budget -1):
            result = homo_ops.homo_rescale(result, cipher_mod_levels, cryptoContext)
        
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN

        digits = hoisting_keyswitch.eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)

        fast_rotation_ext = [None for _ in range(g)]
        
        for j in range(g):
            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = hoisting_keyswitch.eval_fast_rotation_ext(result, digits, rot_in[s][j], True,
                                                                                 cryptoContext)
            else:
                fast_rotation_ext[j] = key_switch_ext(result, True, cryptoContext)
        
        for i in range(b):
            G = g * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[s][G], cryptoContext)
            
            for j in range(1, g):
                if (G + j) != num_rotations:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[s][G + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)
            
            if i == 0:
                first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                F.cv_set_zero(inner.cv[0], len_ext)
                outer = inner
            else:
                if rot_out[s][i] != 0:
                    inner_ks_down = hoisting_keyswitch.key_switch_down(inner, cryptoContext)
                    auto_index = cryptoContext.find_auto_index(rot_out[s][i])

                    first_current = F.cv_automorphism_transform(
                        inner_ks_down.cv[0], curr_limbs, auto_index, cryptoContext)
                    first = cv_add_ext(first, first_current, curr_limbs, cryptoContext)
                    
                    inner_digits = hoisting_keyswitch.eval_fast_rotation_precompute(
                        inner_ks_down.cv[1], inner_ks_down.cur_limbs, cryptoContext
                    )
                    
                    inner_ks_down_ext = hoisting_keyswitch.eval_fast_rotation_ext(inner_ks_down, inner_digits, rot_out[s][i],
                                                                                  False, cryptoContext)
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    tmp_first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                    first = cv_add_ext(first, tmp_first, curr_limbs, cryptoContext)
                    F.cv_set_zero(inner.cv[0], len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)
        
        result = hoisting_keyswitch.key_switch_down(outer, cryptoContext)
        result.cv[0] = cv_add_ext(result.cv[0], first, curr_limbs, cryptoContext)
    
    if flag_rem:
        result = homo_ops.homo_rescale(result, cipher_mod_levels, cryptoContext)
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN

        digits = hoisting_keyswitch.eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)
        
        fast_rotation_ext = [None for _ in range(g_rem)]
        
        s = stop if loop_direction == "backward" else level_budget - flag_rem
        
        for j in range(g_rem):
            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = hoisting_keyswitch.eval_fast_rotation_ext(result, digits, rot_in[s][j],True,
                                                                                 cryptoContext, )
            else:
                fast_rotation_ext[j] = key_switch_ext(result, True, cryptoContext)
        
        for i in range(b_rem):
            G = g_rem * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[s][G], cryptoContext)
            
            for j in range(1, g_rem):
                if (G + j) != num_rotations_rem:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[s][G + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)
            
            if i == 0:
                first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                F.cv_set_zero(inner.cv[0], len_ext)
                outer = inner
            else:
                if rot_out[s][i] != 0:
                    inner_ks_down = hoisting_keyswitch.key_switch_down(inner, cryptoContext)
                    auto_index = cryptoContext.find_auto_index(rot_out[s][i])

                    first_current = F.cv_automorphism_transform(
                        inner_ks_down.cv[0], curr_limbs, auto_index, cryptoContext)
                    first = cv_add_ext(first, first_current, curr_limbs, cryptoContext)
                    
                    inner_digits = hoisting_keyswitch.eval_fast_rotation_precompute(
                        inner_ks_down.cv[1], inner_ks_down.cur_limbs, cryptoContext
                    )
                    
                    inner_ks_down_ext = hoisting_keyswitch.eval_fast_rotation_ext(inner_ks_down, inner_digits, rot_out[s][i], False,
                                                                                  cryptoContext)
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    tmp_first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                    first = cv_add_ext(first, tmp_first, curr_limbs, cryptoContext)
                    F.cv_set_zero(inner.cv[0], len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)
        
        result = hoisting_keyswitch.key_switch_down(outer, cryptoContext)
        result.cv[0] = cv_add_ext(result.cv[0], first, curr_limbs, cryptoContext)
    
    return result



# @profile_python_function
def eval_coeffs_to_slots(A, ctxt, cryptoContext):

    precom = cryptoContext.BsContext

    stop = 0 if precom.paramsEnc.layers_rem != 0 else -1
    flag_rem = 1 if precom.paramsEnc.layers_rem != 0 else 0

    config = {
        "loop_direction": "backward",
        "cipher_mod_levels": 1,
        "eval_fast_rotation_reshape": True,
        "key_switch_ext_size": 2,
        "params": precom.paramsEnc,
        "start": precom.paramsEnc.level_budget - 1,
        "stop": stop
    }

    return merged_function(A, ctxt, cryptoContext, flag_rem, cryptoContext.BsContext.C2S_rot_in, cryptoContext.BsContext.C2S_rot_out, config)

# @profile_python_function
def eval_slots_to_coeffs(A, ctxt, cryptoContext):

    precom = cryptoContext.BsContext
    flag_rem = 1 if precom.paramsDec.layers_rem != 0 else 0

    config = {
        "loop_direction": "forward",
        "cipher_mod_levels": BASE_NUM_LEVELS_TO_DROP,
        "eval_fast_rotation_reshape": False,
        "key_switch_ext_size": NORMAL_CIPHER_SIZE,
        "params": precom.paramsDec,
        "start": 0,
        "stop": precom.paramsDec.level_budget - flag_rem
    }

    return merged_function(A, ctxt, cryptoContext, flag_rem, cryptoContext.BsContext.S2C_rot_in, cryptoContext.BsContext.S2C_rot_out, config)

# @profile_python_function
def eval_linear_transform(A, ct, scheme):
    # TODO: to be implemented
    pass

# @profile_python_function
def cipher_mod_raise(cipher, L0, cryptoContext):
    cv0 = F.cv_switch_modulus_with_intt_ntt(cipher.cv[0], L0, cryptoContext)
    cv1 = F.cv_switch_modulus_with_intt_ntt(cipher.cv[1], L0, cryptoContext)
    return Cipher([cv0, cv1], L0, cipher.scaling_factor, cipher.noise_deg, cipher.slots)

# @profile_python_function
def cipher_mult_by_monomial_and_equal(cipher, monomial_degree, cryptoContext):
    l = cipher.cur_limbs
    cipher.cv[0] = F.cv_mul_by_monomial(cipher.cv[0], l, monomial_degree, cryptoContext)
    cipher.cv[1] = F.cv_mul_by_monomial(cipher.cv[1], l, monomial_degree, cryptoContext)
    return cipher


# @profile_python_function
# note: EvalBootstrap in ckksrns-fhe.cpp
def eval_bootstrap(ciphertext, L0, slots, cryptoContext):
    M = cryptoContext.M
    N = cryptoContext.N
    # cryptoContext.slots = slots #fixme: bad assignment!
    precom = cryptoContext.BsContext
    moduliQ = cryptoContext.moduliQ
    rescaleTech = cryptoContext.rescaleTech

    # note: FLEXIBLEAUTOEXT is not implemented yet
    assert rescaleTech == "FIXEDMANUAL" or rescaleTech == "FLEXIBLEAUTO"

    if (rescaleTech=="FLEXIBLEAUTOEXT"):
        pass
        # For FLEXIBLEAUTOEXT we raised ciphertext does not include extra modulus
        # as it is multiplied by auxiliary plaintext
        #todo: to be implemented, should raise less modulus

    q = moduliQ[0]
    q_double = float(q)

    p = cryptoContext.dcrtBits  # Equivalent to dcrbits in OpenFHE
    powP = 2**p
    deg = utils.round_half_away_from_zero(math.log2(q_double / powP))

    correction = (
        cryptoContext.correctionFactor - deg
    )  # fixme: originally a uint32_t in OpenFHE
    post = 2**deg
    pre = 1.0 / post
    scalar = round(post)

    # -------------------
    # raising the modulus
    # -------------------
    # In FLEXIBLEAUTO, raising the ciphertext to a larger number
    # of towers is a bit more complex, because we need to adjust
    # it's scaling factor to the one that corresponds to the level
    # it's being raised to.
    # Increasing the modulus

    tmp = ciphertext
    tmp = homo_ops.homo_rescale(tmp, tmp.noise_deg-1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)

    # We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    # CKKS bootstrapping faster.
    raised = cipher_mod_raise(tmp, L0, cryptoContext)

    constantEvalMult = pre * (1.0 / (precom.k * N))
    raised = homo_ops.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    ctxtDec = None  # Initialize decrypted ciphertext
    # todo: align with openfhe, but should be refactored. since when only one lb=1, none of them go into EvalLinearTransform.
    isLTBootstrap = (precom.paramsEnc.level_budget == 1) and (precom.paramsDec.level_budget == 1)

    if slots == M // 4: # FULLY PACKED CASE
        # need to call internal modular reduction so it also works for FLEXIBLEAUTO
        raised = homo_ops.homo_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEncI = homo_ops.homo_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)
        ctxtEncI = cipher_mult_by_monomial_and_equal(ctxtEncI, 3 * M // 4, cryptoContext)

        if rescaleTech == "FIXEDMANUAL":
            while(ctxtEnc.noise_deg>1):
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                ctxtEncI = homo_ops.homo_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        else:
            if ctxtEnc.noise_deg==2:
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                ctxtEncI = homo_ops.homo_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------
        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_chebyshev_series_ps(ctxtEnc, precom.coefficients, -1, 1, cryptoContext)
        ctxtEncI = approx.eval_chebyshev_series_ps(ctxtEncI, precom.coefficients, -1, 1, cryptoContext)


        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.homo_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)
        ctxtEncI = apply_double_angle_iterations(ctxtEncI, cryptoContext)

        ctxtEncI = cipher_mult_by_monomial_and_equal(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, ctxtEncI, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------

        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)

    else: # SPARSELY PACKED CASE
        # -------------------
        # Running PartialSum
        # -------------------
        for step in range(int(math.log2(N // (2 * slots)))):
            temp = homo_ops.homo_rotate(raised, (1 << step) * slots, cryptoContext)
            raised = homo_ops.homo_add(raised, temp, cryptoContext)

        # ---------------------
        # Running CoeffsToSlots
        # ---------------------
        raised = homo_ops.homo_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)


        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)

        if rescaleTech == "FIXEDMANUAL":
            while ctxtEnc.noise_deg>1:
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, 1, cryptoContext)
        else:
            if ctxtEnc.noise_deg ==2 :
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, 1, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------

        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_chebyshev_series_ps(ctxtEnc, precom.coefficients, -1, 1, cryptoContext)





        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------
        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)


        ctxtDec_rot = homo_ops.homo_rotate(ctxtDec, slots, cryptoContext)
        ctxtDec = homo_ops.homo_add(ctxtDec, ctxtDec_rot, cryptoContext)





    # 64-bit only: scale back the message to its original scale.
    corFactor = 1 << round(correction)
    ctxtDec = homo_ops.homo_mul_scalar_int(ctxtDec, corFactor, cryptoContext)
    
    return ctxtDec

def homo_bootstrap(cipher, L0, slots, cryptoContext):
    if slots != cipher.slots:
        cp_slots = cipher.slots
        cipher.slots = slots #todo: see if we can remove this
        result = eval_bootstrap(cipher, L0, slots, cryptoContext)
        cipher.slots = cp_slots
    else:
        result = eval_bootstrap(cipher, L0, cipher.slots, cryptoContext)

    if cryptoContext.rescaleTech == "FIXEDMANUAL":  # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
        result = homo_ops.homo_rescale(result, result.noise_deg-1, cryptoContext)

    return result

# def eval_bootstrap_setup(context, level_budget, dim1, numslots, correction_factor):

#     m_U0hatTPreFFT_dim1 = len(context.m_U0hatTPreFFT_dim)
#     m_U0hatTPreFFT_dim2 = context.m_U0hatTPreFFT_dim
#     m_U0hatTPreFFT_limbs = context.m_U0hatTPreFFT_limbs
#     mx_len = context.N
#     mx_slots = numslots
#     m_U0PreFFT_dim1 = len(context.m_U0PreFFT_dim)
#     m_U0PreFFT_dim2 = context.m_U0PreFFT_dim
#     m_U0PreFFT_limbs = context.m_U0PreFFT_limbs

#     M = context.M
#     slots = M // 4 if numslots == 0 else numslots
#     rescale_tech = context.rescaleTech
#     precom = context.BsContext

#     # 设置 correction_factor
#     if correction_factor == 0:
#         if (
#             rescale_tech == "FLEXIBLEAUTO"
#             or rescale_tech == "FLEXIBLEAUTOEXT"
#         ):
#             # 实验结果得出的最佳精度对应的默认 correction factors
#             tmp = utils.round_half_away_from_zero(-0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1)
#             print("inner 2 * math.log2(M / 2)", 2 * math.log2(M / 2))
#             print("inner 2 * math.log2(M / 2) + math.log2(slots)", 2 * math.log2(M / 2) + math.log2(slots))
#             print("inner -0.265 * (2 * math.log2(M / 2) + math.log2(slots))", -0.265 * (2 * math.log2(M / 2) + math.log2(slots)))
#             print("inner -0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1", -0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1)
#             if tmp < 7:
#                 context.correctionFactor = 7
#             elif tmp > 13:
#                 context.correctionFactor = 13
#             else:
#                 context.correctionFactor = int(tmp)
#         else:
#             context.correctionFactor = 9
#     else:
#         context.correctionFactor = correction_factor

#     precom.m_slots = slots
#     precom.m_dim1 = dim1[0]

#     log_slots = math.log2(slots)

#     # 检查 level budget 并计算参数
#     new_budget = [level_budget[0], level_budget[1]]

#     if level_budget[0] > log_slots:
#         print(
#             f"\nWarning, the level budget for encoding cannot be this large. "
#             f"The budget was changed to {int(log_slots)}"
#         )
#         new_budget[0] = int(log_slots)
#     if level_budget[0] < 1:
#         print(
#             f"\nWarning, the level budget for encoding has to be at least 1. "
#             f"The budget was changed to 1"
#         )
#         new_budget[0] = 1

#     if level_budget[1] > log_slots:
#         print(
#             f"\nWarning, the level budget for decoding cannot be this large. "
#             f"The budget was changed to {int(log_slots)}"
#         )
#         new_budget[1] = int(log_slots)
#     if level_budget[1] < 1:
#         print(
#             f"\nWarning, the level budget for decoding has to be at least 1. "
#             f"The budget was changed to 1"
#         )
#         new_budget[1] = 1

#     precom.m_params_enc = context.BsContext.GetCollapsedFFTParams(
#         slots, new_budget[0], dim1[0]
#     )
#     precom.m_params_dec = context.BsContext.GetCollapsedFFTParams(
#         slots, new_budget[1], dim1[1]
#     )

#     if level_budget[0] == 1 and level_budget[1] == 1:
#         pass
#         # # todo: to be implemented, need to get from openfhe
#         # precom.m_U0Pre = [None] * LTMatrix_Row
#         # precom.m_U0hatTPre = [None] * LTMatrix_Row
#         # for i in range(LTMatrix_Row):
#         #     # precom.m_U0hatTPre
#         #     m_U0hatTPre_len = LTMatrix_mx_len * m_U0hatTPre_limbs
#         #     m_U0hatTPre = [m_U0hatTPre_mx[i * m_U0hatTPre_len + j] for j in range(m_U0hatTPre_len)]
#         #     precom.m_U0hatTPre[i] = Plaintext(m_U0hatTPre, LTMatrix_mx_len, LTMatrix_Column, m_U0hatTPre_limbs)
#         #
#         #     # precom.m_U0Pre
#         #     m_U0Pre_len = LTMatrix_mx_len * m_U0Pre_limbs
#         #     m_U0Pre = [m_U0Pre_mx[i * m_U0Pre_len + j] for j in range(m_U0Pre_len)]
#         #     precom.m_U0Pre[i] = Plaintext(m_U0Pre, LTMatrix_mx_len, LTMatrix_Column, m_U0Pre_limbs)
#     else:
#         RHScnt = 0
#         precom.m_U0hatTPreFFT = [[0] * i for i in m_U0hatTPreFFT_dim2]
#         cnt = 0
#         for i in range(0, m_U0hatTPreFFT_dim1):
#             j_len = m_U0hatTPreFFT_dim2[i]
#             limbs = m_U0hatTPreFFT_limbs[i]
#             m_U0hatTPreFFT_len = mx_len * limbs
#             for j in range(j_len):
#                 m_U0hatTPreFFT = np.zeros(m_U0hatTPreFFT_len, dtype=np.uint64)
#                 LHScnt = 0
#                 for k in range(limbs):
#                     for l in range(mx_len):
#                         m_U0hatTPreFFT[LHScnt] = context.m_U0hatTPreFFT_mx[RHScnt]
#                         LHScnt += 1
#                         RHScnt += 1

#                 m_U0hatTPreFFT = torch.tensor(
#                     m_U0hatTPreFFT, dtype=torch.uint64, device="cuda"
#                 )
#                 precom.m_U0hatTPreFFT[i][j] = Plaintext(m_U0hatTPreFFT, mx_len, mx_slots, limbs,
#                                                         context.m_U0hatTPreFFT_scaling_factor[cnt], 1)
#                 cnt+=1

#         cnt=0
#         RHScnt = 0
#         precom.m_U0PreFFT = [[0] * i for i in m_U0PreFFT_dim2]
#         for i in range(m_U0PreFFT_dim1):
#             j_len = m_U0PreFFT_dim2[i]
#             limbs = m_U0PreFFT_limbs[i]
#             m_U0PreFFT_len = mx_len * limbs
#             for j in range(j_len):
#                 m_U0PreFFT = np.zeros(m_U0PreFFT_len, dtype=np.uint64)
#                 LHScnt = 0
#                 for k in range(limbs):
#                     for l in range(mx_len):
#                         m_U0PreFFT[LHScnt] = context.m_U0PreFFT_mx[RHScnt]
#                         LHScnt += 1
#                         RHScnt += 1
#                 m_U0PreFFT = torch.tensor(m_U0PreFFT, dtype=torch.uint64, device="cuda")
#                 precom.m_U0PreFFT[i][j] = Plaintext(m_U0PreFFT, mx_len, mx_slots, limbs,
#                                                     context.m_U0PreFFT_scaling_factor[cnt], 1)
#                 cnt+=1



def BootstrapTest_N65536L26lB44(
    logN=16,
    logSlots=15,
    maxLevelsRemaining=3,
    levelBudget=[4, 4],
    dnum=3,
    dcrtBits=59,
    firstMod=60,
    approxModDepth=9,
    rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
    save_dir="torch/fhe/data/"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    force_update_context = False
    # Force update the context
    if force_update_context:
        gen_contexts(
                logN=logN,
                logSlots=logSlots, # possible slots value of runtime ciphertext #todo: should be a list?
                maxLevelsRemaining=maxLevelsRemaining,
                levelBudget=levelBudget,
                dnum=dnum,
                dcrtBits=dcrtBits,
                firstMod=firstMod,
                approxModDepth=approxModDepth,
                rotate_index=[],
                secretKeyDist="UNIFORM_TERNARY",
                rescaleTech=rescaleTech,
                save_dir=save_dir
            )

    cryptoContext, openfhe_context = utils.try_load_context(logN,
            logSlots,
            maxLevelsRemaining,
            levelBudget,
            dnum,
            dcrtBits,
            firstMod,
            approxModDepth,
            "UNIFORM_TERNARY",
            rescaleTech,
            save_dir=save_dir)


    dim1 = [0, 0]

    # eval_bootstrap_setup(
    #     cryptoContext, cryptoContext.levelBudget, dim1, (1<<logSlots), 0
    # )

    # Test the correctness of the bootstrapping
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<logSlots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1)

    result = eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots), cryptoContext=cryptoContext)
    openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)

    is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
    if is_euqal:
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")

    else:
        print("BootstrapTest_N65536L26lB44: Test failed!")
        print("BootstrapTest_N65536L26lB44: Test failed!")
        print("BootstrapTest_N65536L26lB44: Test failed!")

    exit()

    measure_execution_time = True
    if measure_execution_time:
        start = time.time()
        result = eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots), cryptoContext=cryptoContext)
        end = time.time()
        print("time", end - start)

        # Print the accumulated execution times
        # print("\nTotal execution time for each function:")
        # sorted_execution_times = sorted(utils.execution_times.items(), key=lambda x: x[1], reverse=True)
        # for func_name, total_time in sorted_execution_times:
        #     print(f"{func_name}: {total_time:.6f} seconds")

        pytorch_profiling = False
        if pytorch_profiling:
            # Set up the profiler
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "/home/zrji/log"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:
                # Start profiling specific functions with torch.profiler.record_function()
                result = eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots),
                                        cryptoContext=cryptoContext)

            # Get the profiling results
            profiler_results = profiler.key_averages()

            # Print the profiling summary in a table format
            print(profiler_results.table(sort_by="self_cpu_time_total"))



