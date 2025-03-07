from .bs_context import *
from . import functional as F
from . import homo_ops
from . import approx as approx
from . import hybrid_keyswitch
from . import utils


Tensor = torch.Tensor
NORMAL_CIPHER_SIZE = 2
BASE_NUM_LEVELS_TO_DROP = 1


def adjust_ciphertext(ciphertext, correction, L0, cryptoContext):
    cnst = math.pow(2, -correction)
    ciphertext = homo_ops.homo_mul_scalar_double(ciphertext, cnst, cryptoContext)
    ciphertext = homo_ops.homo_rescale_internal(
        ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext
    )
    return ciphertext


def apply_double_angle_iterations(ciphertext, cryptoContext):
    for j in range(1, 4):
        ciphertext = homo_ops.homo_square(ciphertext, cryptoContext)
        ciphertext = homo_ops.homo_add(ciphertext, ciphertext, cryptoContext)
        scalar = -1.0 / math.pow((2.0 * math.pi), math.pow(2.0, j - 3))
        ciphertext = homo_ops.homo_add_scalar_double(ciphertext, scalar, cryptoContext)
    return ciphertext


def coeffs_slots_conversion(A_Ext, ctxt, direction, cryptoContext):

    if direction == "C2S":
        params = cryptoContext.BsContext.paramsEnc
        rot_in = cryptoContext.BsContext.C2S_rot_in
        rot_out = cryptoContext.BsContext.C2S_rot_out
        loop_range = list(range(0, params.level_budget))[::-1]
    elif direction == "S2C":
        params = cryptoContext.BsContext.paramsDec
        rot_in = cryptoContext.BsContext.S2C_rot_in
        rot_out = cryptoContext.BsContext.S2C_rot_out
        loop_range = list(range(0, params.level_budget))

    num_rotations = params.num_rotations
    b = params.baby_step
    g = params.giant_step

    result = ctxt

    for s in loop_range:
        if not s == loop_range[0]:
            result = homo_ops.homo_rescale_internal(
                result, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )
        if s == loop_range[-1] and params.layers_rem:
            g = params.giant_step_rem
            b = params.baby_step_rem
            num_rotations = params.num_rotations_rem

        digits_ext = hybrid_keyswitch.modup_to_ext(
            homo_ops.extract_cv(result, 1), cryptoContext
        )

        fast_rotation_ext = []

        for j in range(g):
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

        for i in range(b):
            G = g * i
            inner_ext = homo_ops.homo_mul_pt(
                fast_rotation_ext[0], A_Ext[s][G], cryptoContext
            )

            for j in range(1, g):
                if (G + j) != num_rotations:
                    tmp_ext = homo_ops.homo_mul_pt(
                        fast_rotation_ext[j], A_Ext[s][G + j], cryptoContext
                    )
                    inner_ext = homo_ops.homo_add(inner_ext, tmp_ext, cryptoContext)

            if i == 0:
                inner_ext_cv0 = homo_ops.extract_cv(inner_ext, 0)
                first_acc = hybrid_keyswitch.moddown_from_ext(
                    inner_ext_cv0, cryptoContext
                )
                outer_ext = homo_ops.extract_cv(inner_ext, 1, append_zeros=True)
            else:
                if rot_out[s][i] != 0:
                    inner = hybrid_keyswitch.moddown_from_ext(inner_ext, cryptoContext)
                    inner_cv0 = homo_ops.extract_cv(inner, 0)
                    inner_cv1 = homo_ops.extract_cv(inner, 1)

                    first = homo_ops._cipher_automorphism(
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
                    inner_ext_cv0 = homo_ops.extract_cv(inner_ext, 0)
                    first = hybrid_keyswitch.moddown_from_ext(
                        inner_ext_cv0, cryptoContext
                    )
                    first_acc = homo_ops.homo_add(first_acc, first, cryptoContext)
                    inner_ext = homo_ops.extract_cv(inner_ext, 1, append_zeros=True)
                    outer_ext = homo_ops.homo_add(outer_ext, inner_ext, cryptoContext)
        outer = hybrid_keyswitch.moddown_from_ext(outer_ext, cryptoContext)
        first_full_cv = homo_ops.extract_cv(first_acc, 0, append_zeros=True)
        result = homo_ops.homo_add(outer, first_full_cv, cryptoContext)
    return result


def eval_coeffs_to_slots(A, ctxt, cryptoContext):
    return coeffs_slots_conversion(A, ctxt, "C2S", cryptoContext)


def eval_slots_to_coeffs(A, ctxt, cryptoContext):
    return coeffs_slots_conversion(A, ctxt, "S2C", cryptoContext)


def mod_raise(cipher, L0, cryptoContext):
    cv = [
        torch.mod_raise(
            cryptoContext.mod_raise_out,
            cv,
            primes=cryptoContext.primes,
            N=cryptoContext.N,
            L0=L0,
            logN=cryptoContext.logN,
            L=cryptoContext.L,
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


def mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.cur_limbs, monomial_degree, cryptoContext)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.cur_limbs, monomial_degree, cryptoContext)
    return cipher


def eval_bootstrap(ciphertext, L0, logBsSlots, cryptoContext):
    M = cryptoContext.M
    N = cryptoContext.N
    slots = 1 << logBsSlots
    precom = cryptoContext.BsContext
    moduliQ_scalar = cryptoContext.moduliQ_scalar

    q = moduliQ_scalar[0]
    q_double = float(q)

    powP = 2**59
    deg = utils.round_half_away_from_zero(math.log2(q_double / powP))

    if deg > int(precom.correctionFactor):
        print(
            "Warning: Degree [",
            deg,
            "] must be less than or equal to the correction factor[",
            precom.correctionFactor,
            "].",
        )

    correction = precom.correctionFactor - deg
    post = 2**deg
    pre = 1.0 / post
    scalar = round(post)

    tmp = ciphertext
    tmp = homo_ops.homo_rescale_internal(tmp, tmp.noise_deg - 1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)
    raised = mod_raise(tmp, L0, cryptoContext)

    constantEvalMult = pre * (1.0 / (precom.k * N))
    raised = homo_ops.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    ctxtDec = None
    isLTBootstrap = (precom.paramsEnc.level_budget == 1) and (
        precom.paramsDec.level_budget == 1
    )

    if slots == M // 4:
        raised = homo_ops.homo_rescale_internal(
            raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )

        ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEncI = homo_ops.homo_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)
        ctxtEncI = mult_by_monomial_inplace(ctxtEncI, 3 * M // 4, cryptoContext)

        if ctxtEnc.noise_deg == 2:
            ctxtEnc = homo_ops.homo_rescale_internal(
                ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )
            ctxtEncI = homo_ops.homo_rescale_internal(
                ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )

        ctxtEnc = approx.eval_chebyshev_series_ps(
            ctxtEnc, precom.coefficients, -1, 1, cryptoContext
        )
        ctxtEncI = approx.eval_chebyshev_series_ps(
            ctxtEncI, precom.coefficients, -1, 1, cryptoContext
        )

        ctxtEnc = homo_ops.homo_rescale_internal(
            ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )
        ctxtEncI = homo_ops.homo_rescale_internal(
            ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)
        ctxtEncI = apply_double_angle_iterations(ctxtEncI, cryptoContext)
        mult_by_monomial_inplace(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, ctxtEncI, cryptoContext)

        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        ctxtEnc = homo_ops.homo_rescale_internal(
            ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )

        ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)

    else:

        for step in range(int(math.log2(N // (2 * slots)))):
            temp = homo_ops.homo_rotate(raised, (1 << step) * slots, cryptoContext)
            raised = homo_ops.homo_add(raised, temp, cryptoContext)
        raised = homo_ops.homo_rescale_internal(
            raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )

        ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)

        if ctxtEnc.noise_deg == 2:
            ctxtEnc = homo_ops.homo_rescale_internal(ctxtEnc, 1, cryptoContext)

        ctxtEnc = approx.eval_chebyshev_series_ps(
            ctxtEnc, precom.coefficients, -1, 1, cryptoContext
        )
        ctxtEnc = homo_ops.homo_rescale_internal(
            ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        ctxtEnc = homo_ops.homo_rescale_internal(
            ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )

        ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)

        ctxtDec_rot = homo_ops.homo_rotate(ctxtDec, slots, cryptoContext)
        ctxtDec = homo_ops.homo_add(ctxtDec, ctxtDec_rot, cryptoContext)
    corFactor = 1 << round(correction)
    ctxtDec = homo_ops.homo_mul_scalar_int(ctxtDec, corFactor, cryptoContext)

    return ctxtDec


def homo_bootstrap(cipher, L0, logBsSlots, cryptoContext):

    if cryptoContext.autoLoadAndSetConfig == True:
        cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]

    result = eval_bootstrap(cipher, L0, logBsSlots, cryptoContext)

    return result
