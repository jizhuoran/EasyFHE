import time, os
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
        num_towers = ciphertext.cur_limbs
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
#@utils.profile_pytorch_function
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
#@utils.profile_pytorch_function
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
            result = homo_ops.homo_rescale(result, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        if s == loop_range[-1] and params.layers_rem:
            g = params.giant_step_rem
            b = params.baby_step_rem
            num_rotations = params.num_rotations_rem

        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + cryptoContext.K
        len_ext = limbs_ext << cryptoContext.logN

        digits_ext = hoisting_keyswitch.modup_to_ext(result.cipher_like([result.cv[1]]), cryptoContext)

        fast_rotation_ext = []
        
        for j in range(g):
            if rot_in[s][j] != 0:
                fast_rotation_ext.append(hoisting_keyswitch.fused_rotation_add_ext(digits_ext, result, rot_in[s][j],
                                                                                 cryptoContext))
            else:
                fast_rotation_ext.append(hoisting_keyswitch.key_switch_ext(result, cryptoContext))

        # print("times: ", b * g,  b * 2)        
        for i in range(b):
            G = g * i
            inner_ext = hoisting_keyswitch.eval_mult_ext(fast_rotation_ext[0], A_Ext[s][G], cryptoContext)
            for j in range(1, g):
                if (G + j) != num_rotations:
                    tmp_ext = hoisting_keyswitch.eval_mult_ext(fast_rotation_ext[j], A_Ext[s][G + j], cryptoContext)
                    inner_ext = homo_ops.homo_add(inner_ext, tmp_ext, cryptoContext)
            
            if i == 0:
                inner_ext_cv0 = inner_ext.cipher_like([inner_ext.cv[0]])
                first_acc = hoisting_keyswitch.moddown_from_ext(inner_ext_cv0, cryptoContext)
                outer_ext = inner_ext.cipher_like([torch.zeros_like(inner_ext.cv[0]), inner_ext.cv[1]])
            else:
                if rot_out[s][i] != 0:
                    inner = hoisting_keyswitch.moddown_from_ext(inner_ext, cryptoContext)
                    inner_cv0 = inner.cipher_like([inner.cv[0]])
                    inner_cv1 = inner.cipher_like([inner.cv[1]])

                    first = hoisting_keyswitch.eval_automorphism(inner_cv0, rot_out[s][i], cryptoContext)
                    first_acc = homo_ops.homo_add(first_acc, first, cryptoContext)
                    
                    inner_digits = hoisting_keyswitch.modup_to_ext(inner_cv1, cryptoContext)
                    inner_ext = homo_ops.homo_rotate(inner_digits, rot_out[s][i], cryptoContext)
                    outer_ext = homo_ops.homo_add(outer_ext, inner_ext, cryptoContext)
                else:
                    inner_ext_cv0 = inner_ext.cipher_like([inner_ext.cv[0]])
                    first = hoisting_keyswitch.moddown_from_ext(inner_ext_cv0, cryptoContext)
                    first_acc = homo_ops.homo_add(first_acc, first, cryptoContext)
                    inner_ext.cv[0] = torch.zeros_like(inner_ext.cv[0])
                    outer_ext = homo_ops.homo_add(outer_ext, inner_ext, cryptoContext)
        
        outer = hoisting_keyswitch.moddown_from_ext(outer_ext, cryptoContext)
        first_full_cv = first_acc.cipher_like([first_acc.cv[0], torch.zeros_like(first_acc.cv[0])])
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
def mod_raise(cipher, L0, cryptoContext):
    cv0 = F.cv_switch_modulus_with_intt_ntt(cipher.cv[0], L0, cryptoContext)
    cv1 = F.cv_switch_modulus_with_intt_ntt(cipher.cv[1], L0, cryptoContext)
    return Cipher([cv0, cv1], L0, cipher.scaling_factor, cipher.noise_deg, cipher.slots, cipher.is_ext)

# @profile_python_function
def mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.cur_limbs, monomial_degree, cryptoContext, inplace=True)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.cur_limbs, monomial_degree, cryptoContext, inplace=True)

# @profile_python_function
# note: EvalBootstrap in ckksrns-fhe.cpp
#@utils.profile_pytorch_function
def eval_bootstrap(ciphertext, L0, logslots, cryptoContext):
    M = cryptoContext.M
    N = cryptoContext.N
    slots = 1<<logslots
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

    if deg > int(precom.correctionFactor):
        print("Warning: Degree [" , deg ,"] must be less than or equal to the correction factor[",
              precom.correctionFactor, "]." )

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
    tmp = homo_ops.homo_rescale(tmp, tmp.noise_deg-1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)

    # We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    # CKKS bootstrapping faster.
    raised = mod_raise(tmp, L0, cryptoContext)

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
        mult_by_monomial_inplace(ctxtEncI, 3 * M // 4, cryptoContext)

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

def homo_bootstrap(cipher, L0, logSlots, cryptoContext):
    result = eval_bootstrap(cipher, L0, logSlots, cryptoContext)

    if cryptoContext.rescaleTech == "FIXEDMANUAL":  # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
        result = homo_ops.homo_rescale(result, result.noise_deg-1, cryptoContext)

    return result

def BootstrapTest_N65536L26lB44(
    logN=14,
    logSlots_list=[13],
    maxLevelsRemaining=3,
    levelBudget_list=[[4, 4]],
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
                logSlots_list=logSlots_list, # possible slots value of runtime ciphertext #todo: should be a list?
                maxLevelsRemaining=maxLevelsRemaining,
                levelBudget_list=levelBudget_list,
                dnum=dnum,
                dcrtBits=dcrtBits,
                firstMod=firstMod,
                approxModDepth=approxModDepth,
                rotate_index=[],
                secretKeyDist="UNIFORM_TERNARY",
                rescaleTech=rescaleTech,
                save_dir=save_dir
            )

    cryptoContext, openfhe_context_dict = utils.try_load_context(logN,
            logSlots_list,
            maxLevelsRemaining,
            levelBudget_list,
            dnum,
            dcrtBits,
            firstMod,
            approxModDepth,
            [],
            "UNIFORM_TERNARY",
            rescaleTech,
            save_dir=save_dir)

    openfhe_context = openfhe_context_dict[str(logSlots_list[0])]
    dim1 = [0, 0]

    # eval_bootstrap_setup(
    #     cryptoContext, cryptoContext.levelBudget, dim1, (1<<logSlots), 0
    # )

    # Test the correctness of the bootstrapping
    logSlots = logSlots_list[0]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<logSlots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<logSlots)

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(logSlots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, logSlots)
    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)
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
        result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots, cryptoContext=cryptoContext)
        end = time.time()
        print("time", end - start)

        # Print the accumulated execution times
        # print("\nTotal execution time for each function:")
        # sorted_execution_times = sorted(utils.execution_times.items(), key=lambda x: x[1], reverse=True)
        # for func_name, total_time in sorted_execution_times:
        #     print(f"{func_name}: {total_time:.6f} seconds")

        pytorch_profiling = True
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
                result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=logSlots,
                                        cryptoContext=cryptoContext)

            # Get the profiling results
            profiler_results = profiler.key_averages()

            # Print the profiling summary in a table format
            print(profiler_results.table(sort_by="self_cpu_time_total"))


def BootstrapTest_slots_list_example(
        logN=14,
        logSlots_list=[11, 12],
        maxLevelsRemaining=3,
        levelBudget_list=[[3, 3], [4, 4]],
        dnum=3,
        dcrtBits=59,
        firstMod=60,
        approxModDepth=9,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="torch/fhe/data/"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    cryptoContext, openfhe_context_dict = utils.try_load_context(logN,
                                                            logSlots_list,
                                                            maxLevelsRemaining,
                                                            levelBudget_list,
                                                            dnum,
                                                            dcrtBits,
                                                            firstMod,
                                                            approxModDepth,
                                                            [],
                                                            "UNIFORM_TERNARY",
                                                            rescaleTech,
                                                            save_dir=save_dir)

    dim1 = [0, 0]

    # logslots = 11
    specify_slots = logSlots_list[0]
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<specify_slots)

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)
    # utils.load_rotation_keys(cryptoContext, "app") #fixme: deal with "app" is None?

    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    #test correctness
    openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
    is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
    if is_euqal:
        print("BootstrapTest_logslots11: Test passed!")
    else:
        print("BootstrapTest_logslots11: Test failed!")

    # logslots = 12
    specify_slots = logSlots_list[1]
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<specify_slots)

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)

    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    #test correctness
    openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
    is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
    if is_euqal:
        print("BootstrapTest_logslots12: Test passed!")
    else:
        print("BootstrapTest_logslots12: Test failed!")


def BootstrapTest_test_case(
        logN=14,
        logSlots_list=[11, 12],
        maxLevelsRemaining=3,
        levelBudget_list=[[3, 3], [4, 4]],
        dnum=3,
        dcrtBits=59,
        firstMod=60,
        approxModDepth=9,
        rescaleTech = "FLEXIBLEAUTO", # "FLEXIBLEAUTO" # "FIXEDMANUAL"
        save_dir="torch/fhe/data/",
        mode = "debug" # "debug" or "release"

):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist!")

    cryptoContext, openfhe_context_dict = utils.try_load_context(logN,
                                                                 logSlots_list,
                                                                 maxLevelsRemaining,
                                                                 levelBudget_list,
                                                                 dnum,
                                                                 dcrtBits,
                                                                 firstMod,
                                                                 approxModDepth,
                                                                 [-1,2],
                                                                 "UNIFORM_TERNARY",
                                                                 rescaleTech,
                                                                 save_dir=save_dir,
                                                                 mode = mode)

    specify_slots = logSlots_list[0] # logslots = 11
    openfhe_context = openfhe_context_dict[str(specify_slots)]
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<specify_slots))])
    x = torch.tensor(x, device="cuda")
    cipher, cipher_openfhe = openfhe_context.encrypt(x, 1, openfhe_context.depth - 1, 1<<specify_slots)

    # do the application computation
    utils.load_rotation_keys(cryptoContext, "app")
    cipher = homo_ops.homo_rotate(cipher, -1, cryptoContext)
    cipher = homo_ops.homo_rotate(cipher, 2, cryptoContext)
    print("gpu bootstrapp done!")
    # compute golden answer
    if mode == "debug":
        cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe, -1)
        cipher_openfhe = openfhe_context.cc.EvalRotate(cipher_openfhe,2)
        is_euqal = utils.compare_bs_ct_with_openfhe(cipher, cipher_openfhe)
        if is_euqal:
            print("homo_rotate: Test passed!")
        else:
            print("homo_rotate: Test failed!")

    # bootstrapping, logSlots = 11
    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)
    result = eval_bootstrap(cipher, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    print("gpu bootstrapp done!")
    # compute golden answer
    if mode == "debug":
        cipher_openfhe.SetSlots((1<<specify_slots))
        openfhe_boot = openfhe_context.cc.EvalBootstrap(cipher_openfhe)
        is_euqal = utils.compare_bs_ct_with_openfhe(result, openfhe_boot)
        if is_euqal:
            print("BootstrapTest_logslots11: Test passed!")
        else:
            print("BootstrapTest_logslots11: Test failed!")

    # #####################################
    # # ..., omit some homomorphic computation
    # #####################################

    # bootstrapping, logSlots = 12
    specify_slots = logSlots_list[1]
    openfhe_context1 = openfhe_context_dict[str(specify_slots)]

    cryptoContext.BsContext = cryptoContext.BsContext_map[str(specify_slots)]
    cryptoContext.BsContext.to_cuda()
    utils.load_rotation_keys(cryptoContext, specify_slots)
    result1 = eval_bootstrap(result, L0=cryptoContext.L, logslots=specify_slots, cryptoContext=cryptoContext)
    print("gpu bootstrapp done!")
    # compute golden answer
    if mode == "debug":
        openfhe_boot.SetSlots((1 << specify_slots)) # to cheat openfhe boot with (1<<specify_slots)
        openfhe_boot1 = openfhe_context1.cc.EvalBootstrap(openfhe_boot)
        is_euqal = utils.compare_bs_ct_with_openfhe(result1, openfhe_boot1)
        if is_euqal:
            print("BootstrapTest_logslots12: Test passed!")
        else:
            print("BootstrapTest_logslots12: Test failed!")
