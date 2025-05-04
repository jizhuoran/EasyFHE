import math
import torch
from . import functional as F
from . import homo_ops
from . import approx as approx
from . import hybrid_keyswitch
from . import utils
from .dev_tools.decorator_factory import decorator_factory


Tensor = torch.Tensor
NORMAL_CIPHER_SIZE = 2
BASE_NUM_LEVELS_TO_DROP = 1
R_UNIFORM = 6  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
R_SPARSE = 3  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.

@decorator_factory
def assign_scaling_factor(cipher, target_sf, cryptoContext):
    cipher.scaling_factor = target_sf
    return cipher    


def adjust_ciphertext(ciphertext, correction, L0, cryptoContext):
    rescale_tech = cryptoContext.rescaleTech

    if rescale_tech == "FLEXIBLEAUTO" or rescale_tech == "FLEXIBLEAUTOEXT":
        lvl = 0 if rescale_tech == "FLEXIBLEAUTO" else 1
        if cryptoContext.L != L0:
            # Print error message and raise an exception to stop the program
            print("cryptoContext.L != L0")
            raise Exception("Error: cryptoContext.L != L0")
        target_sf = cryptoContext.GetScalingFactorReal(cur_limbs=(L0 - lvl))
        source_sf = ciphertext.scaling_factor
        num_towers = ciphertext.cur_limbs
        mod_to_drop = float(cryptoContext.moduliQ_scalar[num_towers - 1])
        # in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        # a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        # So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        # for NATIVEINT = 128.
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        adjustment_factor = (
                (target_sf / source_sf)
                * (mod_to_drop / source_sf)
                * math.pow(2, -correction)
        )  # if NATIVEINT != 128
        ciphertext = homo_ops.homo_mul_scalar_double(
            ciphertext, adjustment_factor, cryptoContext
        )
        ciphertext = homo_ops.force_rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ciphertext = assign_scaling_factor(ciphertext, target_sf, cryptoContext)

    else:
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        cnst = math.pow(2, -correction)
        ciphertext = homo_ops.homo_mul_scalar_double(ciphertext, cnst, cryptoContext)
        ciphertext = homo_ops.force_rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return ciphertext


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
        ciphertext = homo_ops.homo_rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return ciphertext


def get_collapsed_fft_params(slots: int, level_budget: int):
    log_slots = int(math.log2(slots))
    levels = math.ceil(log_slots / level_budget)
    rem = log_slots % levels

    total_rots = (1 << (levels + 1)) - 1
    shift = 2 if total_rots > 7 else 1
    giant_step = 1 << (levels // 2 + shift)
    baby_step = (total_rots + 1) // giant_step

    if rem:
        rem_rots = (1 << (rem + 1)) - 1
        shift = 2 if rem_rots > 7 else 1
        rem_giant = 1 << (rem // 2 + shift)
        rem_baby = (rem_rots + 1) // rem_giant
    else:
        rem_rots = rem_baby = rem_giant = 0

    return levels, rem, total_rots, baby_step, giant_step, rem_rots, rem_baby, rem_giant


def compute_parameters(slots: int, level_budget: int, M: int, direction: str):
    def reduce_rotation(idx: int, size: int):
        return idx & (size - 1) if (size & (size - 1)) == 0 else idx % size

    levels, rem, total_rots, baby_step, giant_step, rem_rots, rem_baby, rem_giant = \
        get_collapsed_fft_params(slots, level_budget)
    flag_rem = int(rem != 0)

    if direction == 'C2S':
        exp_fn = lambda s: (s - flag_rem) * levels + rem
        loop = range(level_budget - 1, -1 + flag_rem, -1)
        in_div = slots
        rem_index = 0
    else:
        exp_fn = lambda s: s * levels
        loop = range(level_budget - flag_rem)
        in_div = M // 4
        rem_index = level_budget - flag_rem

    center = (total_rots + 1) // 2
    rem_center = (rem_rots + 1) // 2

    rot_in = []
    rot_out = [[0] * (baby_step + rem_baby) for _ in range(level_budget)]

    for i in range(level_budget):
        size = rem_rots + 1 if flag_rem and (
            (direction == 'C2S' and i == 0) or
            (direction == 'S2C' and i == level_budget - 1)
        ) else total_rots + 1
        rot_in.append([0] * size)

    for s in loop:
        exp = exp_fn(s)
        for j in range(giant_step):
            rot_in[s][j] = reduce_rotation((j - center + 1) << exp, in_div)
        for k in range(baby_step):
            rot_out[s][k] = reduce_rotation((giant_step * k) << exp, M // 4)

    if flag_rem:
        s = rem_index
        exp = 0 if direction == 'C2S' else exp_fn(s)
        for j in range(rem_giant):
            rot_in[s][j] = reduce_rotation((j - rem_center + 1) << exp, in_div)
        for k in range(rem_baby):
            rot_out[s][k] = reduce_rotation((rem_giant * k) << exp, M // 4)

    return rem, total_rots, baby_step, giant_step, rem_rots, rem_baby, rem_giant, rot_in, rot_out


def coeffs_slots_conversion(ciphertext, direction: str, slots: int, level_budget: int, cryptoContext):
    rem, total_rots, baby_step, giant_step, rem_rots, rem_baby, rem_giant, rot_in, rot_out = \
        compute_parameters(slots, level_budget, cryptoContext.M, direction)
    num_rotations = total_rots
    loop_range = list(range(level_budget) if direction == 'S2C' else reversed(range(level_budget)))
    result = ciphertext
    logSlots = int(math.log2(slots))

    for s in loop_range:
        if not s == loop_range[0]:
            result = homo_ops.force_rescale(result, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        if s == loop_range[-1] and rem:
            giant_step = rem_giant
            baby_step = rem_baby
            num_rotations = rem_rots

        digits_ext = hybrid_keyswitch.modup_to_ext(
            homo_ops.extract_cv(result, 1, cryptoContext), cryptoContext
        )

        fast_rotation_ext = []

        for j in range(giant_step):
            if rot_in[s][j] != 0:
                fast_rotation_ext.append(
                    homo_ops.eval_fast_rotate(
                        digits_ext, result, rot_in[s][j], True, False, cryptoContext
                    )
                )
            else:
                fast_rotation_ext.append(
                    hybrid_keyswitch.key_switch_P_ext(result, cryptoContext)
                )

        for i in range(baby_step):
            G = giant_step * i
            name = "{}_{}_{}_{}".format(direction, logSlots, s, G)
            inner_ext = homo_ops.homo_mul_pt(
                fast_rotation_ext[0],
                homo_ops.encode(cryptoContext.encode_values[name], name, cryptoContext.L - fast_rotation_ext[0].cur_limbs, cryptoContext.encode_values[name].slots, True, cryptoContext),
                cryptoContext
            )
            for j in range(1, giant_step):
                if (G + j) != num_rotations:
                    name = "{}_{}_{}_{}".format(direction, logSlots, s, G+j)
                    tmp_ext = homo_ops.homo_mul_pt(
                        fast_rotation_ext[j],
                        homo_ops.encode(cryptoContext.encode_values[name], name, cryptoContext.L - fast_rotation_ext[j].cur_limbs, cryptoContext.encode_values[name].slots, True, cryptoContext),
                        cryptoContext
                    )
                    inner_ext = homo_ops.homo_add(inner_ext, tmp_ext, cryptoContext)

            if i == 0:
                inner_ext_cv0 = homo_ops.extract_cv(inner_ext, 0, cryptoContext)
                first_acc = hybrid_keyswitch.moddown_from_ext(
                    inner_ext_cv0, cryptoContext
                )
                outer_ext = homo_ops.extract_cv(inner_ext, 1, cryptoContext, append_zeros=True)
            else:
                if rot_out[s][i] != 0:
                    inner = hybrid_keyswitch.moddown_from_ext(inner_ext, cryptoContext)
                    inner_cv0 = homo_ops.extract_cv(inner, 0, cryptoContext)
                    inner_cv1 = homo_ops.extract_cv(inner, 1, cryptoContext)

                    first = homo_ops.cipher_automorphism(
                        inner_cv0, rot_out[s][i], cryptoContext
                    )
                    first_acc = homo_ops.homo_add(first_acc, first, cryptoContext)

                    inner_digits = hybrid_keyswitch.modup_to_ext(
                        inner_cv1, cryptoContext
                    )
                    inner_ext = homo_ops.eval_fast_rotate(
                        inner_digits, None, rot_out[s][i], False, None, cryptoContext
                    )
                    outer_ext = homo_ops.homo_add(outer_ext, inner_ext, cryptoContext)
                else:
                    inner_ext_cv0 = homo_ops.extract_cv(inner_ext, 0, cryptoContext)
                    first = hybrid_keyswitch.moddown_from_ext(
                        inner_ext_cv0, cryptoContext
                    )
                    first_acc = homo_ops.homo_add(first_acc, first, cryptoContext)
                    inner_ext = homo_ops.extract_cv(inner_ext, 1, cryptoContext, append_zeros=True)
                    outer_ext = homo_ops.homo_add(outer_ext, inner_ext, cryptoContext)
        outer = hybrid_keyswitch.moddown_from_ext(outer_ext, cryptoContext)
        first_full_cv = homo_ops.extract_cv(first_acc, 0, cryptoContext, append_zeros=True)
        result = homo_ops.homo_add(outer, first_full_cv, cryptoContext)
    return result


def eval_coeffs_to_slots(ctxt, slots, level_budget, cryptoContext):
    return coeffs_slots_conversion(ctxt, "C2S", slots, level_budget, cryptoContext)


def eval_slots_to_coeffs(ctxt, slots, level_budget, cryptoContext):
    return coeffs_slots_conversion(ctxt, "S2C", slots, level_budget, cryptoContext)


def eval_linear_transform(ct, scheme):
    # TODO: to be implemented
    pass


@decorator_factory
def mod_raise(cipher, L0, cryptoContext):
    cv = [
        torch.mod_raise(
            cryptoContext.mod_raise_out,
            cv,
            N=cryptoContext.N,
            L0=L0,
            logN=cryptoContext.logN,
            L=cryptoContext.L,
            primes=cryptoContext.primes,
            inverse_power_of_roots_div_two=cryptoContext.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=cryptoContext.inverse_scaled_power_of_roots_div_two,
            power_of_roots_shoup=cryptoContext.power_of_roots_shoup,
            power_of_roots=cryptoContext.power_of_roots,
            barret_ratio=cryptoContext.barret_ratio,
            barret_k=cryptoContext.barret_k,
        ).reshape(-1, cryptoContext.N)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, L0)


@decorator_factory
def mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.cur_limbs, monomial_degree, cryptoContext)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.cur_limbs, monomial_degree, cryptoContext)
    return cipher


# note: EvalBootstrap in ckksrns-fhe.cpp
def eval_bootstrap(ciphertext, L0, logBsSlots, level_budgets, cryptoContext):
    M = cryptoContext.M
    N = cryptoContext.N
    slots = 1 << logBsSlots
    # cryptoContext.slots = slots #fixme: bad assignment!
    # precom = cryptoContext.BsContext
    moduliQ_scalar = cryptoContext.moduliQ_scalar
    rescaleTech = cryptoContext.rescaleTech


    if rescaleTech == "FLEXIBLEAUTOEXT":
        assert False, "FLEXIBLEAUTOEXT is not supported yet."
        # For FLEXIBLEAUTOEXT we raised ciphertext does not include extra modulus
        # as it is multiplied by auxiliary plaintext
        # todo: to be implemented, should raise less modulus

    q = moduliQ_scalar[0]
    q_double = float(q)

    p = cryptoContext.dcrtBits  # Equivalent to dcrbits in OpenFHE
    powP = 2**p
    deg = utils.round_half_away_from_zero(math.log2(q_double / powP))

    if (
        rescaleTech == "FLEXIBLEAUTO"
        or rescaleTech == "FLEXIBLEAUTOEXT"
    ):
        tmp = utils.round_half_away_from_zero(-0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1)
        if tmp < 7:
            correctionFactor = 7
        elif tmp > 13:
            correctionFactor = 13
        else:
            correctionFactor = int(tmp)
    else:
        correctionFactor = 9


    if deg > int(correctionFactor):
        print(
            "Warning: Degree [",
            deg,
            "] must be less than or equal to the correction factor[",
            correctionFactor,
            "].",
        )

    correction = (
            correctionFactor - deg
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
    tmp = homo_ops.force_rescale(tmp, tmp.noise_deg - 1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)

    # We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    # CKKS bootstrapping faster.
    raised = mod_raise(tmp, L0, cryptoContext)

    # In openFHE, K_UNIFORM = 512, K_SPARSE = 28, K_SPARSE is dealt in bs_matrix precompute
    k = 1.0 if cryptoContext.secretKeyDist == "SPARSE_TERNARY" else 512
    constantEvalMult = pre * (1.0 / (k * N))



    raised = homo_ops.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    ctxtDec = None  # Initialize decrypted ciphertext
    # todo: align with openfhe, but should be refactored. since when only one lb=1, none of them go into EvalLinearTransform.
    isLTBootstrap = level_budgets[0] == 1 and level_budgets[1] == 1

    if slots == M // 4:  # FULLY PACKED CASE
        # need to call internal modular reduction so it also works for FLEXIBLEAUTO
        raised = homo_ops.force_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(raised, slots, level_budgets[0], cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEncI = homo_ops.homo_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)
        ctxtEncI = mult_by_monomial_inplace(ctxtEncI, 3 * M // 4, cryptoContext)

        if ctxtEnc.noise_deg == 2: # noise_deg of ctxtEnc and ctxtEncI should be the same
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.force_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------
        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_bootstrapping_chebyshev(
            ctxtEnc, -1, 1, cryptoContext
        )
        ctxtEncI = approx.eval_bootstrapping_chebyshev(
            ctxtEncI, -1, 1, cryptoContext
        )

        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.force_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)
        ctxtEncI = apply_double_angle_iterations(ctxtEncI, cryptoContext)

        mult_by_monomial_inplace(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, ctxtEncI, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------

        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(ctxtEnc, slots, level_budgets[1], cryptoContext)

    else:  # SPARSELY PACKED CASE
        # -------------------
        # Running PartialSum
        # -------------------

        for step in range(int(math.log2(N // (2 * slots)))):
            temp = homo_ops.homo_rotate(raised, (1 << step) * slots, cryptoContext)
            raised = homo_ops.homo_add(raised, temp, cryptoContext)

        # ---------------------
        # Running CoeffsToSlots
        # ---------------------
        raised = homo_ops.force_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(raised, slots, level_budgets[0], cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)
        if ctxtEnc.noise_deg ==2 :
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, 1, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------

        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_bootstrapping_chebyshev(ctxtEnc, -1, 1, cryptoContext)




        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------
        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(ctxtEnc, slots, level_budgets[1], cryptoContext)

        ctxtDec_rot = homo_ops.homo_rotate(ctxtDec, slots, cryptoContext)
        ctxtDec = homo_ops.homo_add(ctxtDec, ctxtDec_rot, cryptoContext)

    # 64-bit only: scale back the message to its original scale.
    corFactor = 1 << round(correction)
    ctxtDec = homo_ops.homo_mul_scalar_int(ctxtDec, corFactor, cryptoContext)

    return ctxtDec


def eval_slim_bootstrap(ciphertext, L0, logBsSlots, level_budgets,cryptoContext):
    M = cryptoContext.M
    N = cryptoContext.N
    slots = 1 << logBsSlots
    # cryptoContext.slots = slots #fixme: bad assignment!
    # precom = cryptoContext.BsContext
    moduliQ_scalar = cryptoContext.moduliQ_scalar
    rescaleTech = cryptoContext.rescaleTech


    if rescaleTech == "FLEXIBLEAUTOEXT":
        assert False, "FLEXIBLEAUTOEXT is not supported yet."
        # For FLEXIBLEAUTOEXT we raised ciphertext does not include extra modulus
        # as it is multiplied by auxiliary plaintext
        # todo: to be implemented, should raise less modulus

    q = moduliQ_scalar[0]
    q_double = float(q)

    p = cryptoContext.dcrtBits  # Equivalent to dcrbits in OpenFHE
    powP = 2**p
    deg = utils.round_half_away_from_zero(math.log2(q_double / powP))

    if (
        rescaleTech == "FLEXIBLEAUTO"
        or rescaleTech == "FLEXIBLEAUTOEXT"
    ):
        tmp = utils.round_half_away_from_zero(-0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1)
        if tmp < 7:
            correctionFactor = 7
        elif tmp > 13:
            correctionFactor = 13
        else:
            correctionFactor = int(tmp)
    else:
        correctionFactor = 9


    if deg > int(correctionFactor):
        print(
            "Warning: Degree [",
            deg,
            "] must be less than or equal to the correction factor[",
            correctionFactor,
            "].",
        )

    correction = (
            correctionFactor - deg
    )  # fixme: originally a uint32_t in OpenFHE
    post = 2**deg
    pre = 1.0 / post
    scalar = round(post)

    ctxtDec = None  # Initialize decrypted ciphertext
    # todo: align with openfhe, but should be refactored. since when only one lb=1, none of them go into EvalLinearTransform.
    isLTBootstrap = level_budgets[0] == 1 and level_budgets[1] == 1

    if slots == M//4:
        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ciphertext, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(ciphertext, slots, level_budgets[1], cryptoContext)
    else:
        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ciphertext, cryptoContext)
        else:
             ctxtDec = eval_slots_to_coeffs(ciphertext, slots, level_budgets[1], cryptoContext)

        ctxtDec_rot = homo_ops.homo_rotate(ctxtDec, slots, cryptoContext)
        ctxtDec = homo_ops.homo_add(ctxtDec, ctxtDec_rot, cryptoContext)


    # -------------------
    # raising the modulus
    # -------------------
    # In FLEXIBLEAUTO, raising the ciphertext to a larger number
    # of towers is a bit more complex, because we need to adjust
    # it's scaling factor to the one that corresponds to the level
    # it's being raised to.
    # Increasing the modulus

    tmp = ctxtDec
    tmp = homo_ops.force_rescale(tmp, tmp.noise_deg - 1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)

    # We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    # CKKS bootstrapping faster.
    raised = mod_raise(tmp, L0, cryptoContext)

    # In openFHE, K_UNIFORM = 512, K_SPARSE = 28, K_SPARSE is dealt in bs_matrix precompute
    k = 1.0 if cryptoContext.secretKeyDist == "SPARSE_TERNARY" else 512
    constantEvalMult = pre * (1.0 / (k * N))
    raised = homo_ops.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    # ctxtDec = None  # Initialize decrypted ciphertext
    # # todo: align with openfhe, but should be refactored. since when only one lb=1, none of them go into EvalLinearTransform.
    # isLTBootstrap = (precom.paramsEnc.level_budget == 1) and (
    #         precom.paramsDec.level_budget == 1
    # )

    if slots == M // 4:  # FULLY PACKED CASE
        # need to call internal modular reduction so it also works for FLEXIBLEAUTO
        raised = homo_ops.force_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(raised, slots, level_budgets[0], cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEncI = homo_ops.homo_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)
        ctxtEncI = mult_by_monomial_inplace(ctxtEncI, 3 * M // 4, cryptoContext)

        if ctxtEnc.noise_deg == 2: # noise_deg of ctxtEnc and ctxtEncI should be the same
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.force_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------
        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_bootstrapping_chebyshev(
            ctxtEnc, -1, 1, cryptoContext
        )
        ctxtEncI = approx.eval_bootstrapping_chebyshev(
            ctxtEncI, -1, 1, cryptoContext
        )

        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.force_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)
        ctxtEncI = apply_double_angle_iterations(ctxtEncI, cryptoContext)

        mult_by_monomial_inplace(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, ctxtEncI, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------

        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        # if isLTBootstrap:
        #     ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        # else:
        #     ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)

    else:  # SPARSELY PACKED CASE
        # -------------------
        # Running PartialSum
        # -------------------

        for step in range(int(math.log2(N // (2 * slots)))):
            temp = homo_ops.homo_rotate(raised, (1 << step) * slots, cryptoContext)
            raised = homo_ops.homo_add(raised, temp, cryptoContext)

        # ---------------------
        # Running CoeffsToSlots
        # ---------------------
        raised = homo_ops.force_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(raised, slots, level_budgets[0], cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)

        if ctxtEnc.noise_deg ==2 :
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, 1, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------

        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_bootstrapping_chebyshev(ctxtEnc, -1, 1, cryptoContext)

        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------
        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.force_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        # if isLTBootstrap:
        #     ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        # else:
        #     ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)
        #
        # ctxtDec_rot = homo_ops.homo_rotate(ctxtDec, slots, cryptoContext)
        # ctxtDec = homo_ops.homo_add(ctxtDec, ctxtDec_rot, cryptoContext)

    # 64-bit only: scale back the message to its original scale.
    corFactor = 1 << round(correction)
    ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, corFactor, cryptoContext)

    return ctxtEnc


@decorator_factory
def homo_bootstrap(cipher, L0, logBsSlots, level_budgets, cryptoContext):

    # cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]

    result = eval_bootstrap(cipher, L0, logBsSlots, level_budgets, cryptoContext)

    # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
    result = homo_ops.homo_rescale(result, result.noise_deg - 1, cryptoContext)

    return result

def homo_double_bootstrap(cipher, L0, logBsSlots, level_budgets, precision, cryptoContext):

    # if cryptoContext.config.AUTO_LOAD_KEYS == True:
    #     cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]

    initSizeQ = cipher.cur_limbs

    # Step 1: Get the input.
    powerOfTwoModulus = 1 << precision

    # Step 2: Scale up by powerOfTwoModulus, and extend the modulus to powerOfTwoModulus * q.
    # Note that we extend the modulus implicitly without any code calls because the value always stays 0.
    ctScaledUp = cipher.deep_copy()
    # We multiply by powerOfTwoModulus, and leave the last CRT value to be 0 (mod powerOfTwoModulus). #todp:??
    ctScaledUp = homo_ops.homo_mul_scalar_int(ctScaledUp, powerOfTwoModulus, cryptoContext)


    # Step 3: Bootstrap the initial ciphertext.
    ctInitialBootstrap = eval_bootstrap(cipher, L0, logBsSlots, level_budgets, cryptoContext)
    ctInitialBootstrap = homo_ops.force_rescale(ctInitialBootstrap, ctInitialBootstrap.noise_deg - 1, cryptoContext)

    # Step 4: Scale up by powerOfTwoModulus.
    ctInitialBootstrap = homo_ops.homo_mul_scalar_int(ctInitialBootstrap, powerOfTwoModulus, cryptoContext)

    # Step 5: Mod-down to powerOfTwoModulus * q
    # We mod down, and leave the last CRT value to be 0 because it's divisible by powerOfTwoModulus.
    ctBootstrappedScaledDown = ctInitialBootstrap.deep_copy()
    bootstrappingSizeQ = ctBootstrappedScaledDown.cur_limbs
    # If we start with more towers, than we obtain from bootstrapping, return the original ciphertext.
    if bootstrappingSizeQ <= initSizeQ:
        return cipher.deep_copy()
    ctBootstrappedScaledDown.cur_limbs = cipher.cur_limbs # note: hard adjust, drop limbs regardless of the rescaleTech


    # Step 6 and 7: Calculate the bootstrapping error by subtracting the original ciphertext from the bootstrapped ciphertext. Mod down to q is done implicitly.
    ctBootstrappingError = homo_ops.homo_sub(ctBootstrappedScaledDown, ctScaledUp, cryptoContext)

    # Step 8: Bootstrap the error.
    ctBootstrappingError = eval_bootstrap(ctBootstrappingError, L0, logBsSlots, level_budgets, cryptoContext)
    ctBootstrappingError = homo_ops.force_rescale(ctBootstrappingError, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

    # Step 9: Subtract the bootstrapped error from the initial bootstrap to get even lower error.
    finalCiphertext = homo_ops.homo_sub(ctInitialBootstrap, ctBootstrappingError, cryptoContext)

    # Step 10: Scale back down by powerOfTwoModulus to get the original message.
    finalCiphertext = homo_ops.homo_mul_scalar_double(finalCiphertext, 1.0 / powerOfTwoModulus, cryptoContext)

    # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
    finalCiphertext = homo_ops.homo_rescale(finalCiphertext, finalCiphertext.noise_deg - 1, cryptoContext)

    return finalCiphertext
