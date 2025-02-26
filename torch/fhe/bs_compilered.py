from .ciphertext import Cipher
from .bs_context import *
from . import functional as F
from . import homo_ops
from . import approx as approx
from . import hybrid_keyswitch
from . import utils


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
        ciphertext = homo_ops.homo_rescale(
            ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )
        ciphertext.scaling_factor = target_sf

    else:
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        cnst = math.pow(2, -correction)
        ciphertext = homo_ops.homo_mul_scalar_double(ciphertext, cnst, cryptoContext)
        ciphertext = homo_ops.homo_rescale(
            ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )
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
        ciphertext = (
            homo_ops.homo_rescale(ciphertext, 1, cryptoContext)
            if cryptoContext.rescaleTech == "FIXEDMANUAL"
            else ciphertext
        )
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

    result = ctxt.deep_copy()

    for s in loop_range:
        if not s == loop_range[0]:
            result = homo_ops.homo_rescale(
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


# @profile_python_function
def eval_coeffs_to_slots(A, ctxt, cryptoContext):
    return coeffs_slots_conversion(A, ctxt, "C2S", cryptoContext)


# @profile_python_function
def eval_slots_to_coeffs(A, ctxt, cryptoContext):
    return coeffs_slots_conversion(A, ctxt, "S2C", cryptoContext)


# @profile_python_function
def eval_linear_transform(A, ct, scheme):
    # TODO: to be implemented
    pass


# @profile_python_function
@utils.printFrontend
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


# @profile_python_function
def mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.cur_limbs, monomial_degree, cryptoContext)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.cur_limbs, monomial_degree, cryptoContext)


# @profile_python_function
# note: EvalBootstrap in ckksrns-fhe.cpp
def eval_bootstrap(NODE_IN, L0, logBsSlots, cryptoContext):
    NODE1 = cryptoContext.BsContext.m_U0hatTPreFFT[0][0]
    NODE2 = cryptoContext.BsContext.m_U0hatTPreFFT[0][1]
    NODE3 = cryptoContext.BsContext.m_U0hatTPreFFT[0][2]
    NODE4 = cryptoContext.BsContext.m_U0hatTPreFFT[0][3]
    NODE5 = cryptoContext.BsContext.m_U0hatTPreFFT[0][4]
    NODE6 = cryptoContext.BsContext.m_U0hatTPreFFT[0][5]
    NODE7 = cryptoContext.BsContext.m_U0hatTPreFFT[0][6]
    NODE8 = cryptoContext.BsContext.m_U0hatTPreFFT[0][7]
    NODE9 = cryptoContext.BsContext.m_U0hatTPreFFT[0][8]
    NODE10 = cryptoContext.BsContext.m_U0hatTPreFFT[0][9]
    NODE11 = cryptoContext.BsContext.m_U0hatTPreFFT[0][10]
    NODE12 = cryptoContext.BsContext.m_U0hatTPreFFT[0][11]
    NODE13 = cryptoContext.BsContext.m_U0hatTPreFFT[0][12]
    NODE14 = cryptoContext.BsContext.m_U0hatTPreFFT[0][13]
    NODE15 = cryptoContext.BsContext.m_U0hatTPreFFT[0][14]
    NODE16 = cryptoContext.BsContext.m_U0hatTPreFFT[1][0]
    NODE17 = cryptoContext.BsContext.m_U0hatTPreFFT[1][1]
    NODE18 = cryptoContext.BsContext.m_U0hatTPreFFT[1][2]
    NODE19 = cryptoContext.BsContext.m_U0hatTPreFFT[1][3]
    NODE20 = cryptoContext.BsContext.m_U0hatTPreFFT[1][4]
    NODE21 = cryptoContext.BsContext.m_U0hatTPreFFT[1][5]
    NODE22 = cryptoContext.BsContext.m_U0hatTPreFFT[1][6]
    NODE23 = cryptoContext.BsContext.m_U0hatTPreFFT[1][7]
    NODE24 = cryptoContext.BsContext.m_U0hatTPreFFT[1][8]
    NODE25 = cryptoContext.BsContext.m_U0hatTPreFFT[1][9]
    NODE26 = cryptoContext.BsContext.m_U0hatTPreFFT[1][10]
    NODE27 = cryptoContext.BsContext.m_U0hatTPreFFT[1][11]
    NODE28 = cryptoContext.BsContext.m_U0hatTPreFFT[1][12]
    NODE29 = cryptoContext.BsContext.m_U0hatTPreFFT[1][13]
    NODE30 = cryptoContext.BsContext.m_U0hatTPreFFT[1][14]
    NODE31 = cryptoContext.BsContext.m_U0hatTPreFFT[2][0]
    NODE32 = cryptoContext.BsContext.m_U0hatTPreFFT[2][1]
    NODE33 = cryptoContext.BsContext.m_U0hatTPreFFT[2][2]
    NODE34 = cryptoContext.BsContext.m_U0hatTPreFFT[2][3]
    NODE35 = cryptoContext.BsContext.m_U0hatTPreFFT[2][4]
    NODE36 = cryptoContext.BsContext.m_U0hatTPreFFT[2][5]
    NODE37 = cryptoContext.BsContext.m_U0hatTPreFFT[2][6]
    NODE38 = cryptoContext.BsContext.m_U0hatTPreFFT[2][7]
    NODE39 = cryptoContext.BsContext.m_U0hatTPreFFT[2][8]
    NODE40 = cryptoContext.BsContext.m_U0hatTPreFFT[2][9]
    NODE41 = cryptoContext.BsContext.m_U0hatTPreFFT[2][10]
    NODE42 = cryptoContext.BsContext.m_U0hatTPreFFT[2][11]
    NODE43 = cryptoContext.BsContext.m_U0hatTPreFFT[2][12]
    NODE44 = cryptoContext.BsContext.m_U0hatTPreFFT[2][13]
    NODE45 = cryptoContext.BsContext.m_U0hatTPreFFT[2][14]
    NODE46 = cryptoContext.BsContext.m_U0hatTPreFFT[3][0]
    NODE47 = cryptoContext.BsContext.m_U0hatTPreFFT[3][1]
    NODE48 = cryptoContext.BsContext.m_U0hatTPreFFT[3][2]
    NODE49 = cryptoContext.BsContext.m_U0hatTPreFFT[3][3]
    NODE50 = cryptoContext.BsContext.m_U0hatTPreFFT[3][4]
    NODE51 = cryptoContext.BsContext.m_U0hatTPreFFT[3][5]
    NODE52 = cryptoContext.BsContext.m_U0hatTPreFFT[3][6]
    NODE53 = cryptoContext.BsContext.m_U0hatTPreFFT[3][7]
    NODE54 = cryptoContext.BsContext.m_U0hatTPreFFT[3][8]
    NODE55 = cryptoContext.BsContext.m_U0hatTPreFFT[3][9]
    NODE56 = cryptoContext.BsContext.m_U0hatTPreFFT[3][10]
    NODE57 = cryptoContext.BsContext.m_U0hatTPreFFT[3][11]
    NODE58 = cryptoContext.BsContext.m_U0hatTPreFFT[3][12]
    NODE59 = cryptoContext.BsContext.m_U0hatTPreFFT[3][13]
    NODE60 = cryptoContext.BsContext.m_U0hatTPreFFT[3][14]
    NODE61 = cryptoContext.BsContext.m_U0PreFFT[0][0]
    NODE62 = cryptoContext.BsContext.m_U0PreFFT[0][1]
    NODE63 = cryptoContext.BsContext.m_U0PreFFT[0][2]
    NODE64 = cryptoContext.BsContext.m_U0PreFFT[0][3]
    NODE65 = cryptoContext.BsContext.m_U0PreFFT[0][4]
    NODE66 = cryptoContext.BsContext.m_U0PreFFT[0][5]
    NODE67 = cryptoContext.BsContext.m_U0PreFFT[0][6]
    NODE68 = cryptoContext.BsContext.m_U0PreFFT[0][7]
    NODE69 = cryptoContext.BsContext.m_U0PreFFT[0][8]
    NODE70 = cryptoContext.BsContext.m_U0PreFFT[0][9]
    NODE71 = cryptoContext.BsContext.m_U0PreFFT[0][10]
    NODE72 = cryptoContext.BsContext.m_U0PreFFT[0][11]
    NODE73 = cryptoContext.BsContext.m_U0PreFFT[0][12]
    NODE74 = cryptoContext.BsContext.m_U0PreFFT[0][13]
    NODE75 = cryptoContext.BsContext.m_U0PreFFT[0][14]
    NODE76 = cryptoContext.BsContext.m_U0PreFFT[1][0]
    NODE77 = cryptoContext.BsContext.m_U0PreFFT[1][1]
    NODE78 = cryptoContext.BsContext.m_U0PreFFT[1][2]
    NODE79 = cryptoContext.BsContext.m_U0PreFFT[1][3]
    NODE80 = cryptoContext.BsContext.m_U0PreFFT[1][4]
    NODE81 = cryptoContext.BsContext.m_U0PreFFT[1][5]
    NODE82 = cryptoContext.BsContext.m_U0PreFFT[1][6]
    NODE83 = cryptoContext.BsContext.m_U0PreFFT[1][7]
    NODE84 = cryptoContext.BsContext.m_U0PreFFT[1][8]
    NODE85 = cryptoContext.BsContext.m_U0PreFFT[1][9]
    NODE86 = cryptoContext.BsContext.m_U0PreFFT[1][10]
    NODE87 = cryptoContext.BsContext.m_U0PreFFT[1][11]
    NODE88 = cryptoContext.BsContext.m_U0PreFFT[1][12]
    NODE89 = cryptoContext.BsContext.m_U0PreFFT[1][13]
    NODE90 = cryptoContext.BsContext.m_U0PreFFT[1][14]
    NODE91 = cryptoContext.BsContext.m_U0PreFFT[2][0]
    NODE92 = cryptoContext.BsContext.m_U0PreFFT[2][1]
    NODE93 = cryptoContext.BsContext.m_U0PreFFT[2][2]
    NODE94 = cryptoContext.BsContext.m_U0PreFFT[2][3]
    NODE95 = cryptoContext.BsContext.m_U0PreFFT[2][4]
    NODE96 = cryptoContext.BsContext.m_U0PreFFT[2][5]
    NODE97 = cryptoContext.BsContext.m_U0PreFFT[2][6]
    NODE98 = cryptoContext.BsContext.m_U0PreFFT[2][7]
    NODE99 = cryptoContext.BsContext.m_U0PreFFT[2][8]
    NODE100 = cryptoContext.BsContext.m_U0PreFFT[2][9]
    NODE101 = cryptoContext.BsContext.m_U0PreFFT[2][10]
    NODE102 = cryptoContext.BsContext.m_U0PreFFT[2][11]
    NODE103 = cryptoContext.BsContext.m_U0PreFFT[2][12]
    NODE104 = cryptoContext.BsContext.m_U0PreFFT[2][13]
    NODE105 = cryptoContext.BsContext.m_U0PreFFT[2][14]
    NODE106 = cryptoContext.BsContext.m_U0PreFFT[3][0]
    NODE107 = cryptoContext.BsContext.m_U0PreFFT[3][1]
    NODE108 = cryptoContext.BsContext.m_U0PreFFT[3][2]
    NODE109 = cryptoContext.BsContext.m_U0PreFFT[3][3]
    NODE110 = cryptoContext.BsContext.m_U0PreFFT[3][4]
    NODE111 = cryptoContext.BsContext.m_U0PreFFT[3][5]
    NODE112 = cryptoContext.BsContext.m_U0PreFFT[3][6]
    NODE113 = cryptoContext.BsContext.m_U0PreFFT[3][7]
    NODE114 = cryptoContext.BsContext.m_U0PreFFT[3][8]
    NODE115 = cryptoContext.BsContext.m_U0PreFFT[3][9]
    NODE116 = cryptoContext.BsContext.m_U0PreFFT[3][10]
    NODE117 = cryptoContext.BsContext.m_U0PreFFT[3][11]
    NODE118 = cryptoContext.BsContext.m_U0PreFFT[3][12]
    NODE119 = cryptoContext.BsContext.m_U0PreFFT[3][13]
    NODE120 = cryptoContext.BsContext.m_U0PreFFT[3][14]

    NODE121 = NODE_IN #my add
    NODE122 = homo_ops.homo_rescale(NODE121, 0, cryptoContext)
    NODE124 = homo_ops.homo_mul_scalar_double(NODE122, 0.00390625, cryptoContext)
    NODE125 = homo_ops.homo_rescale(NODE124, 1, cryptoContext)
    NODE126 = mod_raise(NODE125, 21, cryptoContext)
    NODE127 = homo_ops.homo_mul_scalar_double(NODE126, 3.0517578125e-05, cryptoContext)
    NODE128 = homo_ops.homo_rotate(NODE127, 4096, cryptoContext)
    NODE129 = homo_ops.homo_add(NODE127, NODE128, cryptoContext)
    NODE130 = homo_ops.homo_rescale(NODE129, 1, cryptoContext)
    NODE131 = NODE130.deep_copy()  #my add
    NODE132 = homo_ops.extract_cv(NODE131, 1)
    NODE133 = hybrid_keyswitch.modup_to_ext(NODE132, cryptoContext)
    NODE134 = homo_ops.eval_fast_rotate(NODE133, NODE131, 512, True, False, cryptoContext)
    NODE135 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE133, 512, cryptoContext)
    NODE136 = homo_ops.eval_fast_rotate(NODE133, NODE131, 1024, True, False, cryptoContext)
    NODE137 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE133, 1024, cryptoContext)
    NODE138 = homo_ops.eval_fast_rotate(NODE133, NODE131, 1536, True, False, cryptoContext)
    NODE139 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE133, 1536, cryptoContext)
    NODE140 = homo_ops.eval_fast_rotate(NODE133, NODE131, 2048, True, False, cryptoContext)
    NODE141 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE133, 2048, cryptoContext)
    NODE142 = homo_ops.eval_fast_rotate(NODE133, NODE131, 2560, True, False, cryptoContext)
    NODE143 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE133, 2560, cryptoContext)
    NODE144 = homo_ops.eval_fast_rotate(NODE133, NODE131, 3072, True, False, cryptoContext)
    NODE145 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE133, 3072, cryptoContext)
    NODE146 = homo_ops.eval_fast_rotate(NODE133, NODE131, 3584, True, False, cryptoContext)
    NODE147 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE133, 3584, cryptoContext)
    NODE148 = hybrid_keyswitch.key_switch_P_ext(NODE131, cryptoContext)
    NODE149 = homo_ops.homo_mul_pt(NODE134, NODE46, cryptoContext)
    NODE150 = homo_ops.homo_mul_pt(NODE136, NODE47, cryptoContext)
    NODE151 = homo_ops.homo_add(NODE149, NODE150, cryptoContext)
    NODE152 = homo_ops.homo_mul_pt(NODE138, NODE48, cryptoContext)
    NODE153 = homo_ops.homo_add(NODE151, NODE152, cryptoContext)
    NODE154 = homo_ops.homo_mul_pt(NODE140, NODE49, cryptoContext)
    NODE155 = homo_ops.homo_add(NODE153, NODE154, cryptoContext)
    NODE156 = homo_ops.homo_mul_pt(NODE142, NODE50, cryptoContext)
    NODE157 = homo_ops.homo_add(NODE155, NODE156, cryptoContext)
    NODE158 = homo_ops.homo_mul_pt(NODE144, NODE51, cryptoContext)
    NODE159 = homo_ops.homo_add(NODE157, NODE158, cryptoContext)
    NODE160 = homo_ops.homo_mul_pt(NODE146, NODE52, cryptoContext)
    NODE161 = homo_ops.homo_add(NODE159, NODE160, cryptoContext)
    NODE162 = homo_ops.homo_mul_pt(NODE148, NODE53, cryptoContext)
    NODE163 = homo_ops.homo_add(NODE161, NODE162, cryptoContext)
    NODE164 = homo_ops.extract_cv(NODE163, 0)
    NODE165 = hybrid_keyswitch.moddown_from_ext(NODE164, cryptoContext)
    NODE166 = homo_ops.extract_cv(NODE163, 1, append_zeros = True)
    NODE167 = homo_ops.homo_mul_pt(NODE134, NODE54, cryptoContext)
    NODE168 = homo_ops.homo_mul_pt(NODE136, NODE55, cryptoContext)
    NODE169 = homo_ops.homo_add(NODE167, NODE168, cryptoContext)
    NODE170 = homo_ops.homo_mul_pt(NODE138, NODE56, cryptoContext)
    NODE171 = homo_ops.homo_add(NODE169, NODE170, cryptoContext)
    NODE172 = homo_ops.homo_mul_pt(NODE140, NODE57, cryptoContext)
    NODE173 = homo_ops.homo_add(NODE171, NODE172, cryptoContext)
    NODE174 = homo_ops.homo_mul_pt(NODE142, NODE58, cryptoContext)
    NODE175 = homo_ops.homo_add(NODE173, NODE174, cryptoContext)
    NODE176 = homo_ops.homo_mul_pt(NODE144, NODE59, cryptoContext)
    NODE177 = homo_ops.homo_add(NODE175, NODE176, cryptoContext)
    NODE178 = homo_ops.homo_mul_pt(NODE146, NODE60, cryptoContext)
    NODE179 = homo_ops.homo_add(NODE177, NODE178, cryptoContext)
    NODE180 = hybrid_keyswitch.moddown_from_ext(NODE179, cryptoContext)
    NODE181 = homo_ops.extract_cv(NODE180, 0)
    NODE182 = homo_ops.extract_cv(NODE180, 1)
    NODE183 = homo_ops._cipher_automorphism(NODE181, 4096, cryptoContext)
    NODE184 = homo_ops.homo_add(NODE165, NODE183, cryptoContext)
    NODE185 = hybrid_keyswitch.modup_to_ext(NODE182, cryptoContext)
    NODE186 = homo_ops.eval_fast_rotate(NODE185, None, 4096, False, None, cryptoContext)
    NODE187 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE185, 4096, cryptoContext)
    NODE188 = homo_ops.homo_add(NODE166, NODE186, cryptoContext)
    NODE189 = hybrid_keyswitch.moddown_from_ext(NODE188, cryptoContext)
    NODE190 = homo_ops.extract_cv(NODE184, 0, append_zeros = True)
    NODE191 = homo_ops.homo_add(NODE189, NODE190, cryptoContext)
    NODE192 = homo_ops.homo_rescale(NODE191, 1, cryptoContext)
    NODE193 = homo_ops.extract_cv(NODE192, 1)
    NODE194 = hybrid_keyswitch.modup_to_ext(NODE193, cryptoContext)
    NODE195 = homo_ops.eval_fast_rotate(NODE194, NODE192, 3648, True, False, cryptoContext)
    NODE196 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE194, 3648, cryptoContext)
    NODE197 = homo_ops.eval_fast_rotate(NODE194, NODE192, 3712, True, False, cryptoContext)
    NODE198 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE194, 3712, cryptoContext)
    NODE199 = homo_ops.eval_fast_rotate(NODE194, NODE192, 3776, True, False, cryptoContext)
    NODE200 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE194, 3776, cryptoContext)
    NODE201 = homo_ops.eval_fast_rotate(NODE194, NODE192, 3840, True, False, cryptoContext)
    NODE202 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE194, 3840, cryptoContext)
    NODE203 = homo_ops.eval_fast_rotate(NODE194, NODE192, 3904, True, False, cryptoContext)
    NODE204 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE194, 3904, cryptoContext)
    NODE205 = homo_ops.eval_fast_rotate(NODE194, NODE192, 3968, True, False, cryptoContext)
    NODE206 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE194, 3968, cryptoContext)
    NODE207 = homo_ops.eval_fast_rotate(NODE194, NODE192, 4032, True, False, cryptoContext)
    NODE208 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE194, 4032, cryptoContext)
    NODE209 = hybrid_keyswitch.key_switch_P_ext(NODE192, cryptoContext)
    NODE210 = homo_ops.homo_mul_pt(NODE195, NODE31, cryptoContext)
    NODE211 = homo_ops.homo_mul_pt(NODE197, NODE32, cryptoContext)
    NODE212 = homo_ops.homo_add(NODE210, NODE211, cryptoContext)
    NODE213 = homo_ops.homo_mul_pt(NODE199, NODE33, cryptoContext)
    NODE214 = homo_ops.homo_add(NODE212, NODE213, cryptoContext)
    NODE215 = homo_ops.homo_mul_pt(NODE201, NODE34, cryptoContext)
    NODE216 = homo_ops.homo_add(NODE214, NODE215, cryptoContext)
    NODE217 = homo_ops.homo_mul_pt(NODE203, NODE35, cryptoContext)
    NODE218 = homo_ops.homo_add(NODE216, NODE217, cryptoContext)
    NODE219 = homo_ops.homo_mul_pt(NODE205, NODE36, cryptoContext)
    NODE220 = homo_ops.homo_add(NODE218, NODE219, cryptoContext)
    NODE221 = homo_ops.homo_mul_pt(NODE207, NODE37, cryptoContext)
    NODE222 = homo_ops.homo_add(NODE220, NODE221, cryptoContext)
    NODE223 = homo_ops.homo_mul_pt(NODE209, NODE38, cryptoContext)
    NODE224 = homo_ops.homo_add(NODE222, NODE223, cryptoContext)
    NODE225 = homo_ops.extract_cv(NODE224, 0)
    NODE226 = hybrid_keyswitch.moddown_from_ext(NODE225, cryptoContext)
    NODE227 = homo_ops.extract_cv(NODE224, 1, append_zeros = True)
    NODE228 = homo_ops.homo_mul_pt(NODE195, NODE39, cryptoContext)
    NODE229 = homo_ops.homo_mul_pt(NODE197, NODE40, cryptoContext)
    NODE230 = homo_ops.homo_add(NODE228, NODE229, cryptoContext)
    NODE231 = homo_ops.homo_mul_pt(NODE199, NODE41, cryptoContext)
    NODE232 = homo_ops.homo_add(NODE230, NODE231, cryptoContext)
    NODE233 = homo_ops.homo_mul_pt(NODE201, NODE42, cryptoContext)
    NODE234 = homo_ops.homo_add(NODE232, NODE233, cryptoContext)
    NODE235 = homo_ops.homo_mul_pt(NODE203, NODE43, cryptoContext)
    NODE236 = homo_ops.homo_add(NODE234, NODE235, cryptoContext)
    NODE237 = homo_ops.homo_mul_pt(NODE205, NODE44, cryptoContext)
    NODE238 = homo_ops.homo_add(NODE236, NODE237, cryptoContext)
    NODE239 = homo_ops.homo_mul_pt(NODE207, NODE45, cryptoContext)
    NODE240 = homo_ops.homo_add(NODE238, NODE239, cryptoContext)
    NODE241 = hybrid_keyswitch.moddown_from_ext(NODE240, cryptoContext)
    NODE242 = homo_ops.extract_cv(NODE241, 0)
    NODE243 = homo_ops.extract_cv(NODE241, 1)
    NODE244 = homo_ops._cipher_automorphism(NODE242, 512, cryptoContext)
    NODE245 = homo_ops.homo_add(NODE226, NODE244, cryptoContext)
    NODE246 = hybrid_keyswitch.modup_to_ext(NODE243, cryptoContext)
    NODE247 = homo_ops.eval_fast_rotate(NODE246, None, 512, False, None, cryptoContext)
    NODE248 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE246, 512, cryptoContext)
    NODE249 = homo_ops.homo_add(NODE227, NODE247, cryptoContext)
    NODE250 = hybrid_keyswitch.moddown_from_ext(NODE249, cryptoContext)
    NODE251 = homo_ops.extract_cv(NODE245, 0, append_zeros = True)
    NODE252 = homo_ops.homo_add(NODE250, NODE251, cryptoContext)
    NODE253 = homo_ops.homo_rescale(NODE252, 1, cryptoContext)
    NODE254 = homo_ops.extract_cv(NODE253, 1)
    NODE255 = hybrid_keyswitch.modup_to_ext(NODE254, cryptoContext)
    NODE256 = homo_ops.eval_fast_rotate(NODE255, NODE253, 4040, True, False, cryptoContext)
    NODE257 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE255, 4040, cryptoContext)
    NODE258 = homo_ops.eval_fast_rotate(NODE255, NODE253, 4048, True, False, cryptoContext)
    NODE259 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE255, 4048, cryptoContext)
    NODE260 = homo_ops.eval_fast_rotate(NODE255, NODE253, 4056, True, False, cryptoContext)
    NODE261 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE255, 4056, cryptoContext)
    NODE262 = homo_ops.eval_fast_rotate(NODE255, NODE253, 4064, True, False, cryptoContext)
    NODE263 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE255, 4064, cryptoContext)
    NODE264 = homo_ops.eval_fast_rotate(NODE255, NODE253, 4072, True, False, cryptoContext)
    NODE265 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE255, 4072, cryptoContext)
    NODE266 = homo_ops.eval_fast_rotate(NODE255, NODE253, 4080, True, False, cryptoContext)
    NODE267 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE255, 4080, cryptoContext)
    NODE268 = homo_ops.eval_fast_rotate(NODE255, NODE253, 4088, True, False, cryptoContext)
    NODE269 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE255, 4088, cryptoContext)
    NODE270 = hybrid_keyswitch.key_switch_P_ext(NODE253, cryptoContext)
    NODE271 = homo_ops.homo_mul_pt(NODE256, NODE16, cryptoContext)
    NODE272 = homo_ops.homo_mul_pt(NODE258, NODE17, cryptoContext)
    NODE273 = homo_ops.homo_add(NODE271, NODE272, cryptoContext)
    NODE274 = homo_ops.homo_mul_pt(NODE260, NODE18, cryptoContext)
    NODE275 = homo_ops.homo_add(NODE273, NODE274, cryptoContext)
    NODE276 = homo_ops.homo_mul_pt(NODE262, NODE19, cryptoContext)
    NODE277 = homo_ops.homo_add(NODE275, NODE276, cryptoContext)
    NODE278 = homo_ops.homo_mul_pt(NODE264, NODE20, cryptoContext)
    NODE279 = homo_ops.homo_add(NODE277, NODE278, cryptoContext)
    NODE280 = homo_ops.homo_mul_pt(NODE266, NODE21, cryptoContext)
    NODE281 = homo_ops.homo_add(NODE279, NODE280, cryptoContext)
    NODE282 = homo_ops.homo_mul_pt(NODE268, NODE22, cryptoContext)
    NODE283 = homo_ops.homo_add(NODE281, NODE282, cryptoContext)
    NODE284 = homo_ops.homo_mul_pt(NODE270, NODE23, cryptoContext)
    NODE285 = homo_ops.homo_add(NODE283, NODE284, cryptoContext)
    NODE286 = homo_ops.extract_cv(NODE285, 0)
    NODE287 = hybrid_keyswitch.moddown_from_ext(NODE286, cryptoContext)
    NODE288 = homo_ops.extract_cv(NODE285, 1, append_zeros = True)
    NODE289 = homo_ops.homo_mul_pt(NODE256, NODE24, cryptoContext)
    NODE290 = homo_ops.homo_mul_pt(NODE258, NODE25, cryptoContext)
    NODE291 = homo_ops.homo_add(NODE289, NODE290, cryptoContext)
    NODE292 = homo_ops.homo_mul_pt(NODE260, NODE26, cryptoContext)
    NODE293 = homo_ops.homo_add(NODE291, NODE292, cryptoContext)
    NODE294 = homo_ops.homo_mul_pt(NODE262, NODE27, cryptoContext)
    NODE295 = homo_ops.homo_add(NODE293, NODE294, cryptoContext)
    NODE296 = homo_ops.homo_mul_pt(NODE264, NODE28, cryptoContext)
    NODE297 = homo_ops.homo_add(NODE295, NODE296, cryptoContext)
    NODE298 = homo_ops.homo_mul_pt(NODE266, NODE29, cryptoContext)
    NODE299 = homo_ops.homo_add(NODE297, NODE298, cryptoContext)
    NODE300 = homo_ops.homo_mul_pt(NODE268, NODE30, cryptoContext)
    NODE301 = homo_ops.homo_add(NODE299, NODE300, cryptoContext)
    NODE302 = hybrid_keyswitch.moddown_from_ext(NODE301, cryptoContext)
    NODE303 = homo_ops.extract_cv(NODE302, 0)
    NODE304 = homo_ops.extract_cv(NODE302, 1)
    NODE305 = homo_ops._cipher_automorphism(NODE303, 64, cryptoContext)
    NODE306 = homo_ops.homo_add(NODE287, NODE305, cryptoContext)
    NODE307 = hybrid_keyswitch.modup_to_ext(NODE304, cryptoContext)
    NODE308 = homo_ops.eval_fast_rotate(NODE307, None, 64, False, None, cryptoContext)
    NODE309 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE307, 64, cryptoContext)
    NODE310 = homo_ops.homo_add(NODE288, NODE308, cryptoContext)
    NODE311 = hybrid_keyswitch.moddown_from_ext(NODE310, cryptoContext)
    NODE312 = homo_ops.extract_cv(NODE306, 0, append_zeros = True)
    NODE313 = homo_ops.homo_add(NODE311, NODE312, cryptoContext)
    NODE314 = homo_ops.homo_rescale(NODE313, 1, cryptoContext)
    NODE315 = homo_ops.extract_cv(NODE314, 1)
    NODE316 = hybrid_keyswitch.modup_to_ext(NODE315, cryptoContext)
    NODE317 = homo_ops.eval_fast_rotate(NODE316, NODE314, 4089, True, False, cryptoContext)
    NODE318 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE316, 4089, cryptoContext)
    NODE319 = homo_ops.eval_fast_rotate(NODE316, NODE314, 4090, True, False, cryptoContext)
    NODE320 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE316, 4090, cryptoContext)
    NODE321 = homo_ops.eval_fast_rotate(NODE316, NODE314, 4091, True, False, cryptoContext)
    NODE322 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE316, 4091, cryptoContext)
    NODE323 = homo_ops.eval_fast_rotate(NODE316, NODE314, 4092, True, False, cryptoContext)
    NODE324 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE316, 4092, cryptoContext)
    NODE325 = homo_ops.eval_fast_rotate(NODE316, NODE314, 4093, True, False, cryptoContext)
    NODE326 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE316, 4093, cryptoContext)
    NODE327 = homo_ops.eval_fast_rotate(NODE316, NODE314, 4094, True, False, cryptoContext)
    NODE328 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE316, 4094, cryptoContext)
    NODE329 = homo_ops.eval_fast_rotate(NODE316, NODE314, 4095, True, False, cryptoContext)
    NODE330 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE316, 4095, cryptoContext)
    NODE331 = hybrid_keyswitch.key_switch_P_ext(NODE314, cryptoContext)
    NODE332 = homo_ops.homo_mul_pt(NODE317, NODE1, cryptoContext)
    NODE333 = homo_ops.homo_mul_pt(NODE319, NODE2, cryptoContext)
    NODE334 = homo_ops.homo_add(NODE332, NODE333, cryptoContext)
    NODE335 = homo_ops.homo_mul_pt(NODE321, NODE3, cryptoContext)
    NODE336 = homo_ops.homo_add(NODE334, NODE335, cryptoContext)
    NODE337 = homo_ops.homo_mul_pt(NODE323, NODE4, cryptoContext)
    NODE338 = homo_ops.homo_add(NODE336, NODE337, cryptoContext)
    NODE339 = homo_ops.homo_mul_pt(NODE325, NODE5, cryptoContext)
    NODE340 = homo_ops.homo_add(NODE338, NODE339, cryptoContext)
    NODE341 = homo_ops.homo_mul_pt(NODE327, NODE6, cryptoContext)
    NODE342 = homo_ops.homo_add(NODE340, NODE341, cryptoContext)
    NODE343 = homo_ops.homo_mul_pt(NODE329, NODE7, cryptoContext)
    NODE344 = homo_ops.homo_add(NODE342, NODE343, cryptoContext)
    NODE345 = homo_ops.homo_mul_pt(NODE331, NODE8, cryptoContext)
    NODE346 = homo_ops.homo_add(NODE344, NODE345, cryptoContext)
    NODE347 = homo_ops.extract_cv(NODE346, 0)
    NODE348 = hybrid_keyswitch.moddown_from_ext(NODE347, cryptoContext)
    NODE349 = homo_ops.extract_cv(NODE346, 1, append_zeros = True)
    NODE350 = homo_ops.homo_mul_pt(NODE317, NODE9, cryptoContext)
    NODE351 = homo_ops.homo_mul_pt(NODE319, NODE10, cryptoContext)
    NODE352 = homo_ops.homo_add(NODE350, NODE351, cryptoContext)
    NODE353 = homo_ops.homo_mul_pt(NODE321, NODE11, cryptoContext)
    NODE354 = homo_ops.homo_add(NODE352, NODE353, cryptoContext)
    NODE355 = homo_ops.homo_mul_pt(NODE323, NODE12, cryptoContext)
    NODE356 = homo_ops.homo_add(NODE354, NODE355, cryptoContext)
    NODE357 = homo_ops.homo_mul_pt(NODE325, NODE13, cryptoContext)
    NODE358 = homo_ops.homo_add(NODE356, NODE357, cryptoContext)
    NODE359 = homo_ops.homo_mul_pt(NODE327, NODE14, cryptoContext)
    NODE360 = homo_ops.homo_add(NODE358, NODE359, cryptoContext)
    NODE361 = homo_ops.homo_mul_pt(NODE329, NODE15, cryptoContext)
    NODE362 = homo_ops.homo_add(NODE360, NODE361, cryptoContext)
    NODE363 = hybrid_keyswitch.moddown_from_ext(NODE362, cryptoContext)
    NODE364 = homo_ops.extract_cv(NODE363, 0)
    NODE365 = homo_ops.extract_cv(NODE363, 1)
    NODE366 = homo_ops._cipher_automorphism(NODE364, 8, cryptoContext)
    NODE367 = homo_ops.homo_add(NODE348, NODE366, cryptoContext)
    NODE368 = hybrid_keyswitch.modup_to_ext(NODE365, cryptoContext)
    NODE369 = homo_ops.eval_fast_rotate(NODE368, None, 8, False, None, cryptoContext)
    NODE370 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE368, 8, cryptoContext)
    NODE371 = homo_ops.homo_add(NODE349, NODE369, cryptoContext)
    NODE372 = hybrid_keyswitch.moddown_from_ext(NODE371, cryptoContext)
    NODE373 = homo_ops.extract_cv(NODE367, 0, append_zeros = True)
    NODE374 = homo_ops.homo_add(NODE372, NODE373, cryptoContext)
    NODE375 = homo_ops.homo_rotate(NODE374, 32767, cryptoContext)
    NODE376 = homo_ops.homo_add(NODE374, NODE375, cryptoContext)
    NODE377 = homo_ops.homo_rescale(NODE376, 1, cryptoContext)

    # ---------------------------------
    # Running Approximate Mod Reduction
    # ---------------------------------

    # Evaluate Chebyshev series for the sine wave
    ctxtEnc = approx.eval_chebyshev_series_ps(
        NODE377, cryptoContext.BsContext.coefficients, -1, 1, cryptoContext
    )
    return ctxtEnc



    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        ctxtEnc = homo_ops.homo_rescale(
            ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
        )
    ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

    NODE495 = homo_ops.homo_mul_scalar_int(ctxtEnc, 2, cryptoContext)
    NODE496 = NODE495.deep_copy() #my add
    NODE497 = homo_ops.extract_cv(NODE496, 1)
    NODE498 = hybrid_keyswitch.modup_to_ext(NODE497, cryptoContext)
    NODE499 = homo_ops.eval_fast_rotate(NODE498, NODE496, 8185, True, False, cryptoContext)
    NODE500 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE498, 8185, cryptoContext)
    NODE501 = homo_ops.eval_fast_rotate(NODE498, NODE496, 8186, True, False, cryptoContext)
    NODE502 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE498, 8186, cryptoContext)
    NODE503 = homo_ops.eval_fast_rotate(NODE498, NODE496, 8187, True, False, cryptoContext)
    NODE504 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE498, 8187, cryptoContext)
    NODE505 = homo_ops.eval_fast_rotate(NODE498, NODE496, 8188, True, False, cryptoContext)
    NODE506 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE498, 8188, cryptoContext)
    NODE507 = homo_ops.eval_fast_rotate(NODE498, NODE496, 8189, True, False, cryptoContext)
    NODE508 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE498, 8189, cryptoContext)
    NODE509 = homo_ops.eval_fast_rotate(NODE498, NODE496, 8190, True, False, cryptoContext)
    NODE510 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE498, 8190, cryptoContext)
    NODE511 = homo_ops.eval_fast_rotate(NODE498, NODE496, 8191, True, False, cryptoContext)
    NODE512 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE498, 8191, cryptoContext)
    NODE513 = hybrid_keyswitch.key_switch_P_ext(NODE496, cryptoContext)
    NODE514 = homo_ops.homo_mul_pt(NODE499, NODE61, cryptoContext)
    NODE515 = homo_ops.homo_mul_pt(NODE501, NODE62, cryptoContext)
    NODE516 = homo_ops.homo_add(NODE514, NODE515, cryptoContext)
    NODE517 = homo_ops.homo_mul_pt(NODE503, NODE63, cryptoContext)
    NODE518 = homo_ops.homo_add(NODE516, NODE517, cryptoContext)
    NODE519 = homo_ops.homo_mul_pt(NODE505, NODE64, cryptoContext)
    NODE520 = homo_ops.homo_add(NODE518, NODE519, cryptoContext)
    NODE521 = homo_ops.homo_mul_pt(NODE507, NODE65, cryptoContext)
    NODE522 = homo_ops.homo_add(NODE520, NODE521, cryptoContext)
    NODE523 = homo_ops.homo_mul_pt(NODE509, NODE66, cryptoContext)
    NODE524 = homo_ops.homo_add(NODE522, NODE523, cryptoContext)
    NODE525 = homo_ops.homo_mul_pt(NODE511, NODE67, cryptoContext)
    NODE526 = homo_ops.homo_add(NODE524, NODE525, cryptoContext)
    NODE527 = homo_ops.homo_mul_pt(NODE513, NODE68, cryptoContext)
    NODE528 = homo_ops.homo_add(NODE526, NODE527, cryptoContext)
    NODE529 = homo_ops.extract_cv(NODE528, 0)
    NODE530 = hybrid_keyswitch.moddown_from_ext(NODE529, cryptoContext)
    NODE531 = homo_ops.extract_cv(NODE528, 1, append_zeros = True)
    NODE532 = homo_ops.homo_mul_pt(NODE499, NODE69, cryptoContext)
    NODE533 = homo_ops.homo_mul_pt(NODE501, NODE70, cryptoContext)
    NODE534 = homo_ops.homo_add(NODE532, NODE533, cryptoContext)
    NODE535 = homo_ops.homo_mul_pt(NODE503, NODE71, cryptoContext)
    NODE536 = homo_ops.homo_add(NODE534, NODE535, cryptoContext)
    NODE537 = homo_ops.homo_mul_pt(NODE505, NODE72, cryptoContext)
    NODE538 = homo_ops.homo_add(NODE536, NODE537, cryptoContext)
    NODE539 = homo_ops.homo_mul_pt(NODE507, NODE73, cryptoContext)
    NODE540 = homo_ops.homo_add(NODE538, NODE539, cryptoContext)
    NODE541 = homo_ops.homo_mul_pt(NODE509, NODE74, cryptoContext)
    NODE542 = homo_ops.homo_add(NODE540, NODE541, cryptoContext)
    NODE543 = homo_ops.homo_mul_pt(NODE511, NODE75, cryptoContext)
    NODE544 = homo_ops.homo_add(NODE542, NODE543, cryptoContext)
    NODE545 = hybrid_keyswitch.moddown_from_ext(NODE544, cryptoContext)
    NODE546 = homo_ops.extract_cv(NODE545, 0)
    NODE547 = homo_ops.extract_cv(NODE545, 1)
    NODE548 = homo_ops._cipher_automorphism(NODE546, 8, cryptoContext)
    NODE549 = homo_ops.homo_add(NODE530, NODE548, cryptoContext)
    NODE550 = hybrid_keyswitch.modup_to_ext(NODE547, cryptoContext)
    NODE551 = homo_ops.eval_fast_rotate(NODE550, None, 8, False, None, cryptoContext)
    NODE552 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE550, 8, cryptoContext)
    NODE553 = homo_ops.homo_add(NODE531, NODE551, cryptoContext)
    NODE554 = hybrid_keyswitch.moddown_from_ext(NODE553, cryptoContext)
    NODE555 = homo_ops.extract_cv(NODE549, 0, append_zeros = True)
    NODE556 = homo_ops.homo_add(NODE554, NODE555, cryptoContext)
    NODE557 = homo_ops.homo_rescale(NODE556, 1, cryptoContext)
    NODE558 = homo_ops.extract_cv(NODE557, 1)
    NODE559 = hybrid_keyswitch.modup_to_ext(NODE558, cryptoContext)
    NODE560 = homo_ops.eval_fast_rotate(NODE559, NODE557, 8136, True, False, cryptoContext)
    NODE561 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE559, 8136, cryptoContext)
    NODE562 = homo_ops.eval_fast_rotate(NODE559, NODE557, 8144, True, False, cryptoContext)
    NODE563 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE559, 8144, cryptoContext)
    NODE564 = homo_ops.eval_fast_rotate(NODE559, NODE557, 8152, True, False, cryptoContext)
    NODE565 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE559, 8152, cryptoContext)
    NODE566 = homo_ops.eval_fast_rotate(NODE559, NODE557, 8160, True, False, cryptoContext)
    NODE567 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE559, 8160, cryptoContext)
    NODE568 = homo_ops.eval_fast_rotate(NODE559, NODE557, 8168, True, False, cryptoContext)
    NODE569 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE559, 8168, cryptoContext)
    NODE570 = homo_ops.eval_fast_rotate(NODE559, NODE557, 8176, True, False, cryptoContext)
    NODE571 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE559, 8176, cryptoContext)
    NODE572 = homo_ops.eval_fast_rotate(NODE559, NODE557, 8184, True, False, cryptoContext)
    NODE573 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE559, 8184, cryptoContext)
    NODE574 = hybrid_keyswitch.key_switch_P_ext(NODE557, cryptoContext)
    NODE575 = homo_ops.homo_mul_pt(NODE560, NODE76, cryptoContext)
    NODE576 = homo_ops.homo_mul_pt(NODE562, NODE77, cryptoContext)
    NODE577 = homo_ops.homo_add(NODE575, NODE576, cryptoContext)
    NODE578 = homo_ops.homo_mul_pt(NODE564, NODE78, cryptoContext)
    NODE579 = homo_ops.homo_add(NODE577, NODE578, cryptoContext)
    NODE580 = homo_ops.homo_mul_pt(NODE566, NODE79, cryptoContext)
    NODE581 = homo_ops.homo_add(NODE579, NODE580, cryptoContext)
    NODE582 = homo_ops.homo_mul_pt(NODE568, NODE80, cryptoContext)
    NODE583 = homo_ops.homo_add(NODE581, NODE582, cryptoContext)
    NODE584 = homo_ops.homo_mul_pt(NODE570, NODE81, cryptoContext)
    NODE585 = homo_ops.homo_add(NODE583, NODE584, cryptoContext)
    NODE586 = homo_ops.homo_mul_pt(NODE572, NODE82, cryptoContext)
    NODE587 = homo_ops.homo_add(NODE585, NODE586, cryptoContext)
    NODE588 = homo_ops.homo_mul_pt(NODE574, NODE83, cryptoContext)
    NODE589 = homo_ops.homo_add(NODE587, NODE588, cryptoContext)
    NODE590 = homo_ops.extract_cv(NODE589, 0)
    NODE591 = hybrid_keyswitch.moddown_from_ext(NODE590, cryptoContext)
    NODE592 = homo_ops.extract_cv(NODE589, 1, append_zeros = True)
    NODE593 = homo_ops.homo_mul_pt(NODE560, NODE84, cryptoContext)
    NODE594 = homo_ops.homo_mul_pt(NODE562, NODE85, cryptoContext)
    NODE595 = homo_ops.homo_add(NODE593, NODE594, cryptoContext)
    NODE596 = homo_ops.homo_mul_pt(NODE564, NODE86, cryptoContext)
    NODE597 = homo_ops.homo_add(NODE595, NODE596, cryptoContext)
    NODE598 = homo_ops.homo_mul_pt(NODE566, NODE87, cryptoContext)
    NODE599 = homo_ops.homo_add(NODE597, NODE598, cryptoContext)
    NODE600 = homo_ops.homo_mul_pt(NODE568, NODE88, cryptoContext)
    NODE601 = homo_ops.homo_add(NODE599, NODE600, cryptoContext)
    NODE602 = homo_ops.homo_mul_pt(NODE570, NODE89, cryptoContext)
    NODE603 = homo_ops.homo_add(NODE601, NODE602, cryptoContext)
    NODE604 = homo_ops.homo_mul_pt(NODE572, NODE90, cryptoContext)
    NODE605 = homo_ops.homo_add(NODE603, NODE604, cryptoContext)
    NODE606 = hybrid_keyswitch.moddown_from_ext(NODE605, cryptoContext)
    NODE607 = homo_ops.extract_cv(NODE606, 0)
    NODE608 = homo_ops.extract_cv(NODE606, 1)
    NODE609 = homo_ops._cipher_automorphism(NODE607, 64, cryptoContext)
    NODE610 = homo_ops.homo_add(NODE591, NODE609, cryptoContext)
    NODE611 = hybrid_keyswitch.modup_to_ext(NODE608, cryptoContext)
    NODE612 = homo_ops.eval_fast_rotate(NODE611, None, 64, False, None, cryptoContext)
    NODE613 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE611, 64, cryptoContext)
    NODE614 = homo_ops.homo_add(NODE592, NODE612, cryptoContext)
    NODE615 = hybrid_keyswitch.moddown_from_ext(NODE614, cryptoContext)
    NODE616 = homo_ops.extract_cv(NODE610, 0, append_zeros = True)
    NODE617 = homo_ops.homo_add(NODE615, NODE616, cryptoContext)
    NODE618 = homo_ops.homo_rescale(NODE617, 1, cryptoContext)
    NODE619 = homo_ops.extract_cv(NODE618, 1)
    NODE620 = hybrid_keyswitch.modup_to_ext(NODE619, cryptoContext)
    NODE621 = homo_ops.eval_fast_rotate(NODE620, NODE618, 7744, True, False, cryptoContext)
    NODE622 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE620, 7744, cryptoContext)
    NODE623 = homo_ops.eval_fast_rotate(NODE620, NODE618, 7808, True, False, cryptoContext)
    NODE624 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE620, 7808, cryptoContext)
    NODE625 = homo_ops.eval_fast_rotate(NODE620, NODE618, 7872, True, False, cryptoContext)
    NODE626 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE620, 7872, cryptoContext)
    NODE627 = homo_ops.eval_fast_rotate(NODE620, NODE618, 7936, True, False, cryptoContext)
    NODE628 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE620, 7936, cryptoContext)
    NODE629 = homo_ops.eval_fast_rotate(NODE620, NODE618, 8000, True, False, cryptoContext)
    NODE630 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE620, 8000, cryptoContext)
    NODE631 = homo_ops.eval_fast_rotate(NODE620, NODE618, 8064, True, False, cryptoContext)
    NODE632 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE620, 8064, cryptoContext)
    NODE633 = homo_ops.eval_fast_rotate(NODE620, NODE618, 8128, True, False, cryptoContext)
    NODE634 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE620, 8128, cryptoContext)
    NODE635 = hybrid_keyswitch.key_switch_P_ext(NODE618, cryptoContext)
    NODE636 = homo_ops.homo_mul_pt(NODE621, NODE91, cryptoContext)
    NODE637 = homo_ops.homo_mul_pt(NODE623, NODE92, cryptoContext)
    NODE638 = homo_ops.homo_add(NODE636, NODE637, cryptoContext)
    NODE639 = homo_ops.homo_mul_pt(NODE625, NODE93, cryptoContext)
    NODE640 = homo_ops.homo_add(NODE638, NODE639, cryptoContext)
    NODE641 = homo_ops.homo_mul_pt(NODE627, NODE94, cryptoContext)
    NODE642 = homo_ops.homo_add(NODE640, NODE641, cryptoContext)
    NODE643 = homo_ops.homo_mul_pt(NODE629, NODE95, cryptoContext)
    NODE644 = homo_ops.homo_add(NODE642, NODE643, cryptoContext)
    NODE645 = homo_ops.homo_mul_pt(NODE631, NODE96, cryptoContext)
    NODE646 = homo_ops.homo_add(NODE644, NODE645, cryptoContext)
    NODE647 = homo_ops.homo_mul_pt(NODE633, NODE97, cryptoContext)
    NODE648 = homo_ops.homo_add(NODE646, NODE647, cryptoContext)
    NODE649 = homo_ops.homo_mul_pt(NODE635, NODE98, cryptoContext)
    NODE650 = homo_ops.homo_add(NODE648, NODE649, cryptoContext)
    NODE651 = homo_ops.extract_cv(NODE650, 0)
    NODE652 = hybrid_keyswitch.moddown_from_ext(NODE651, cryptoContext)
    NODE653 = homo_ops.extract_cv(NODE650, 1, append_zeros = True)
    NODE654 = homo_ops.homo_mul_pt(NODE621, NODE99, cryptoContext)
    NODE655 = homo_ops.homo_mul_pt(NODE623, NODE100, cryptoContext)
    NODE656 = homo_ops.homo_add(NODE654, NODE655, cryptoContext)
    NODE657 = homo_ops.homo_mul_pt(NODE625, NODE101, cryptoContext)
    NODE658 = homo_ops.homo_add(NODE656, NODE657, cryptoContext)
    NODE659 = homo_ops.homo_mul_pt(NODE627, NODE102, cryptoContext)
    NODE660 = homo_ops.homo_add(NODE658, NODE659, cryptoContext)
    NODE661 = homo_ops.homo_mul_pt(NODE629, NODE103, cryptoContext)
    NODE662 = homo_ops.homo_add(NODE660, NODE661, cryptoContext)
    NODE663 = homo_ops.homo_mul_pt(NODE631, NODE104, cryptoContext)
    NODE664 = homo_ops.homo_add(NODE662, NODE663, cryptoContext)
    NODE665 = homo_ops.homo_mul_pt(NODE633, NODE105, cryptoContext)
    NODE666 = homo_ops.homo_add(NODE664, NODE665, cryptoContext)
    NODE667 = hybrid_keyswitch.moddown_from_ext(NODE666, cryptoContext)
    NODE668 = homo_ops.extract_cv(NODE667, 0)
    NODE669 = homo_ops.extract_cv(NODE667, 1)
    NODE670 = homo_ops._cipher_automorphism(NODE668, 512, cryptoContext)
    NODE671 = homo_ops.homo_add(NODE652, NODE670, cryptoContext)
    NODE672 = hybrid_keyswitch.modup_to_ext(NODE669, cryptoContext)
    NODE673 = homo_ops.eval_fast_rotate(NODE672, None, 512, False, None, cryptoContext)
    NODE674 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE672, 512, cryptoContext)
    NODE675 = homo_ops.homo_add(NODE653, NODE673, cryptoContext)
    NODE676 = hybrid_keyswitch.moddown_from_ext(NODE675, cryptoContext)
    NODE677 = homo_ops.extract_cv(NODE671, 0, append_zeros = True)
    NODE678 = homo_ops.homo_add(NODE676, NODE677, cryptoContext)
    NODE679 = homo_ops.homo_rescale(NODE678, 1, cryptoContext)
    NODE680 = homo_ops.extract_cv(NODE679, 1)
    NODE681 = hybrid_keyswitch.modup_to_ext(NODE680, cryptoContext)
    NODE682 = homo_ops.eval_fast_rotate(NODE681, NODE679, 4608, True, False, cryptoContext)
    NODE683 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE681, 4608, cryptoContext)
    NODE684 = homo_ops.eval_fast_rotate(NODE681, NODE679, 5120, True, False, cryptoContext)
    NODE685 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE681, 5120, cryptoContext)
    NODE686 = homo_ops.eval_fast_rotate(NODE681, NODE679, 5632, True, False, cryptoContext)
    NODE687 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE681, 5632, cryptoContext)
    NODE688 = homo_ops.eval_fast_rotate(NODE681, NODE679, 6144, True, False, cryptoContext)
    NODE689 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE681, 6144, cryptoContext)
    NODE690 = homo_ops.eval_fast_rotate(NODE681, NODE679, 6656, True, False, cryptoContext)
    NODE691 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE681, 6656, cryptoContext)
    NODE692 = homo_ops.eval_fast_rotate(NODE681, NODE679, 7168, True, False, cryptoContext)
    NODE693 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE681, 7168, cryptoContext)
    NODE694 = homo_ops.eval_fast_rotate(NODE681, NODE679, 7680, True, False, cryptoContext)
    NODE695 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE681, 7680, cryptoContext)
    NODE696 = hybrid_keyswitch.key_switch_P_ext(NODE679, cryptoContext)
    NODE697 = homo_ops.homo_mul_pt(NODE682, NODE106, cryptoContext)
    NODE698 = homo_ops.homo_mul_pt(NODE684, NODE107, cryptoContext)
    NODE699 = homo_ops.homo_add(NODE697, NODE698, cryptoContext)
    NODE700 = homo_ops.homo_mul_pt(NODE686, NODE108, cryptoContext)
    NODE701 = homo_ops.homo_add(NODE699, NODE700, cryptoContext)
    NODE702 = homo_ops.homo_mul_pt(NODE688, NODE109, cryptoContext)
    NODE703 = homo_ops.homo_add(NODE701, NODE702, cryptoContext)
    NODE704 = homo_ops.homo_mul_pt(NODE690, NODE110, cryptoContext)
    NODE705 = homo_ops.homo_add(NODE703, NODE704, cryptoContext)
    NODE706 = homo_ops.homo_mul_pt(NODE692, NODE111, cryptoContext)
    NODE707 = homo_ops.homo_add(NODE705, NODE706, cryptoContext)
    NODE708 = homo_ops.homo_mul_pt(NODE694, NODE112, cryptoContext)
    NODE709 = homo_ops.homo_add(NODE707, NODE708, cryptoContext)
    NODE710 = homo_ops.homo_mul_pt(NODE696, NODE113, cryptoContext)
    NODE711 = homo_ops.homo_add(NODE709, NODE710, cryptoContext)
    NODE712 = homo_ops.extract_cv(NODE711, 0)
    NODE713 = hybrid_keyswitch.moddown_from_ext(NODE712, cryptoContext)
    NODE714 = homo_ops.extract_cv(NODE711, 1, append_zeros = True)
    NODE715 = homo_ops.homo_mul_pt(NODE682, NODE114, cryptoContext)
    NODE716 = homo_ops.homo_mul_pt(NODE684, NODE115, cryptoContext)
    NODE717 = homo_ops.homo_add(NODE715, NODE716, cryptoContext)
    NODE718 = homo_ops.homo_mul_pt(NODE686, NODE116, cryptoContext)
    NODE719 = homo_ops.homo_add(NODE717, NODE718, cryptoContext)
    NODE720 = homo_ops.homo_mul_pt(NODE688, NODE117, cryptoContext)
    NODE721 = homo_ops.homo_add(NODE719, NODE720, cryptoContext)
    NODE722 = homo_ops.homo_mul_pt(NODE690, NODE118, cryptoContext)
    NODE723 = homo_ops.homo_add(NODE721, NODE722, cryptoContext)
    NODE724 = homo_ops.homo_mul_pt(NODE692, NODE119, cryptoContext)
    NODE725 = homo_ops.homo_add(NODE723, NODE724, cryptoContext)
    NODE726 = homo_ops.homo_mul_pt(NODE694, NODE120, cryptoContext)
    NODE727 = homo_ops.homo_add(NODE725, NODE726, cryptoContext)
    NODE728 = hybrid_keyswitch.moddown_from_ext(NODE727, cryptoContext)
    NODE729 = homo_ops.extract_cv(NODE728, 0)
    NODE730 = homo_ops.extract_cv(NODE728, 1)
    NODE731 = homo_ops._cipher_automorphism(NODE729, 4096, cryptoContext)
    NODE732 = homo_ops.homo_add(NODE713, NODE731, cryptoContext)
    NODE733 = hybrid_keyswitch.modup_to_ext(NODE730, cryptoContext)
    NODE734 = homo_ops.eval_fast_rotate(NODE733, None, 4096, False, None, cryptoContext)
    NODE735 = hybrid_keyswitch.mult_rot_key_and_sum_ext(NODE733, 4096, cryptoContext)
    NODE736 = homo_ops.homo_add(NODE714, NODE734, cryptoContext)
    NODE737 = hybrid_keyswitch.moddown_from_ext(NODE736, cryptoContext)
    NODE738 = homo_ops.extract_cv(NODE732, 0, append_zeros = True)
    NODE739 = homo_ops.homo_add(NODE737, NODE738, cryptoContext)
    NODE740 = homo_ops.homo_rotate(NODE739, 4096, cryptoContext)
    NODE741 = homo_ops.homo_add(NODE739, NODE740, cryptoContext)
    NODE742 = homo_ops.homo_mul_scalar_int(NODE741, 256, cryptoContext)
    return NODE742
    
    precom = cryptoContext.BsContext
    ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, NODE10, cryptoContext)
    NODE158 = ctxtEnc
    NODE159 = homo_ops.homo_rotate(NODE158, 32767, cryptoContext)
    NODE160 = homo_ops.homo_add(NODE158, NODE159, cryptoContext)
    NODE161 = homo_ops.homo_rescale(NODE160, 1, cryptoContext)





    M = cryptoContext.M
    N = cryptoContext.N
    slots = 1 << logBsSlots
    # cryptoContext.slots = slots #fixme: bad assignment!
    precom = cryptoContext.BsContext
    moduliQ_scalar = cryptoContext.moduliQ_scalar
    rescaleTech = cryptoContext.rescaleTech

    # note: FLEXIBLEAUTOEXT is not implemented yet
    assert rescaleTech == "FIXEDMANUAL" or rescaleTech == "FLEXIBLEAUTO"

    if rescaleTech == "FLEXIBLEAUTOEXT":
        pass
        # For FLEXIBLEAUTOEXT we raised ciphertext does not include extra modulus
        # as it is multiplied by auxiliary plaintext
        # todo: to be implemented, should raise less modulus

    q = moduliQ_scalar[0]
    q_double = float(q)

    p = cryptoContext.dcrtBits  # Equivalent to dcrbits in OpenFHE
    powP = 2**p
    deg = utils.round_half_away_from_zero(math.log2(q_double / powP))

    if deg > int(precom.correctionFactor):
        print(
            "Warning: Degree [",
            deg,
            "] must be less than or equal to the correction factor[",
            precom.correctionFactor,
            "].",
        )

    correction = (
        precom.correctionFactor - deg
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
    tmp = homo_ops.homo_rescale(tmp, tmp.noise_deg - 1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)

    # We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    # CKKS bootstrapping faster.
    raised = mod_raise(tmp, L0, cryptoContext)

    constantEvalMult = pre * (1.0 / (precom.k * N))
    raised = homo_ops.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    ctxtDec = None  # Initialize decrypted ciphertext
    # todo: align with openfhe, but should be refactored. since when only one lb=1, none of them go into EvalLinearTransform.
    isLTBootstrap = (precom.paramsEnc.level_budget == 1) and (
        precom.paramsDec.level_budget == 1
    )

    if slots == M // 4:  # FULLY PACKED CASE
        # need to call internal modular reduction so it also works for FLEXIBLEAUTO
        raised = homo_ops.homo_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEncI = homo_ops.homo_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)
        mult_by_monomial_inplace(ctxtEncI, 3 * M // 4, cryptoContext)

        if rescaleTech == "FIXEDMANUAL":
            while ctxtEnc.noise_deg > 1:
                ctxtEnc = homo_ops.homo_rescale(
                    ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
                )
                ctxtEncI = homo_ops.homo_rescale(
                    ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext
                )
        else:
            if ctxtEnc.noise_deg == 2:
                ctxtEnc = homo_ops.homo_rescale(
                    ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
                )
                ctxtEncI = homo_ops.homo_rescale(
                    ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext
                )

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------
        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_chebyshev_series_ps(
            ctxtEnc, precom.coefficients, -1, 1, cryptoContext
        )
        ctxtEncI = approx.eval_chebyshev_series_ps(
            ctxtEncI, precom.coefficients, -1, 1, cryptoContext
        )

        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(
                ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )
            ctxtEncI = homo_ops.homo_rescale(
                ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )
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
            ctxtEnc = homo_ops.homo_rescale(
                ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)

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
        raised = homo_ops.homo_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        exit(0)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)

        if rescaleTech == "FIXEDMANUAL":
            while ctxtEnc.noise_deg > 1:
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, 1, cryptoContext)
        else:
            if ctxtEnc.noise_deg == 2:
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, 1, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------

        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = approx.eval_chebyshev_series_ps(
            ctxtEnc, precom.coefficients, -1, 1, cryptoContext
        )

        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(
                ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------
        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(
                ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )

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


def homo_bootstrap(cipher, L0, logBsSlots, cryptoContext):

    if cryptoContext.autoLoadAndSetConfig == True:
        cryptoContext.BsContext = cryptoContext.BsContext_map[str(logBsSlots)]

    result = eval_bootstrap(cipher, L0, logBsSlots, cryptoContext)

    if (
        cryptoContext.rescaleTech == "FIXEDMANUAL"
    ):  # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
        result = homo_ops.homo_rescale(result, result.noise_deg - 1, cryptoContext)

    return result
