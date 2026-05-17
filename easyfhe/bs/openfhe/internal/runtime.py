import math
import easyfhe as torch
from easyfhe.fhe.ops import kernels as F
from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import homo
from easyfhe.fhe.ops import rotation
from . import approx as bootstrap_approx


Tensor = torch.Tensor
NORMAL_CIPHER_SIZE = 2
BASE_NUM_LEVELS_TO_DROP = 1


def _match_state(ciphertext, cur_limbs, noise_deg, scaling_factor, cryptoContext):
    return alignment.align_to(
        ciphertext,
        alignment.CipherState(cur_limbs, noise_deg, scaling_factor),
        cryptoContext,
    )


def _rescale(ciphertext, levels, cryptoContext):
    return _match_state(
        ciphertext,
        ciphertext.cur_limbs - levels,
        ciphertext.noise_deg - levels,
        None,
        cryptoContext,
    )

def assign_scaling_factor(cipher, target_sf, cryptoContext):
    return _assign_scaling_factor(cipher, target_sf, cryptoContext)


def _assign_scaling_factor(cipher, target_sf, cryptoContext):
    cipher.scaling_factor = target_sf
    return cipher


def adjust_ciphertext(ciphertext, correction, L0, cryptoContext):
    rescale_tech = cryptoContext.rescaleTech

    if rescale_tech == "FLEXIBLEAUTO":
        lvl = 0
        if cryptoContext.L != L0:
            # Print error message and raise an exception to stop the program
            print("cryptoContext.L != L0")
            raise Exception("Error: cryptoContext.L != L0")
        target_sf = cryptoContext.scale_at(cur_limbs=(L0 - lvl))
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
        ciphertext = homo.homo_mul_scalar_double(
            ciphertext, adjustment_factor, cryptoContext
        )
        ciphertext = _rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ciphertext = assign_scaling_factor(ciphertext, target_sf, cryptoContext)

    else:
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        cnst = math.pow(2, -correction)
        ciphertext = homo.homo_mul_scalar_double(ciphertext, cnst, cryptoContext)
        ciphertext = _rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return ciphertext


def _validate_transform_plan(plan, constants):
    if constants is None:
        raise ValueError(f"{plan.direction} conversion requires explicit bootstrap constants")


def _constant_info(constants, name):
    try:
        return constants.info[name]
    except AttributeError:
        return getattr(constants, name)


def _constant_scalar(constants, name):
    try:
        scalar_name = constants.info.get("scalar_names", {}).get(name, name)
        return constants.scalar(scalar_name)
    except AttributeError:
        return getattr(constants, name)


def _homo_conjugate(cipher, cryptoContext):
    return homo.homo_rotate(cipher, 2 * cryptoContext.N - 1, cryptoContext)


def coeffs_slots_conversion(ciphertext, transform_plan, constants, cryptoContext):
    _validate_transform_plan(transform_plan, constants)
    plan = transform_plan
    result = ciphertext
    strategy = _constant_info(constants, "strategy")
    hoist_strategy = {
        "normal_bsgs": "normal",
        "normal_giant": "ext_normal",
    }.get(strategy, "ext_double_hoist")
    is_ext = strategy != "normal_bsgs"
    batch_specs = constants.info["plaintext_batches"][plan.direction]

    for loop_pos, level in enumerate(plan.loop_range):
        if loop_pos != 0:
            result = _rescale(result, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if loop_pos == len(plan.loop_range) - 1 and plan.rem:
            giant_step = int(plan.giant_step_rem)
            baby_step = int(plan.baby_step_rem)
        else:
            giant_step = int(plan.giant_step)
            baby_step = int(plan.baby_step)

        spec = batch_specs[int(level)]
        if int(spec["baby_step"]) != baby_step or int(spec["giant_step"]) != giant_step:
            raise ValueError(
                f"{plan.direction} plaintext batch metadata mismatch at level {level}: "
                f"got baby={spec['baby_step']} giant={spec['giant_step']}, "
                f"expected baby={baby_step} giant={giant_step}"
            )

        giant_offsets = plan.rot_out[level][:baby_step]
        giant_offset = int(giant_offsets[1]) - int(giant_offsets[0]) if len(giant_offsets) > 1 else 0
        plaintext_batch = constants.plaintext_batch(
            spec["names"],
            cryptoContext.L - result.cur_limbs,
            spec["slots"],
            cryptoContext,
            is_ext=is_ext,
        )
        result = rotation.hoisted_mac_sum(
            result,
            plan.rot_in[level][:giant_step],
            plaintext_batch,
            giant_offset,
            baby_step,
            cryptoContext,
            strategy=hoist_strategy,
        )
    return result


def eval_coeffs_to_slots(ctxt, cryptoContext, bootstrap_constants):
    return coeffs_slots_conversion(
        ctxt,
        _constant_info(bootstrap_constants, "c2s_plan"),
        bootstrap_constants,
        cryptoContext,
    )


def eval_slots_to_coeffs(ctxt, cryptoContext, bootstrap_constants):
    return coeffs_slots_conversion(
        ctxt,
        _constant_info(bootstrap_constants, "s2c_plan"),
        bootstrap_constants,
        cryptoContext,
    )


def eval_linear_transform(ct, scheme):
    # TODO: to be implemented
    pass


def mod_raise(cipher, L0, cryptoContext):
    return _mod_raise(cipher, L0, cryptoContext)


def _mod_raise(cipher, L0, cryptoContext):
    cv = [
        torch.mod_raise(
            cv.reshape(1, 1, cv.shape[0], cv.shape[1]),
            N=cryptoContext.N,
            L0=L0,
            old_prime=cryptoContext.primes_list[0],
            primes=cryptoContext.primes,
            switch_modulus_map=cryptoContext.switch_modulus_map,
            inverse_power_of_roots_div_two=cryptoContext.inverse_power_of_roots_div_two,
            inverse_scaled_power_of_roots_div_two=cryptoContext.inverse_scaled_power_of_roots_div_two,
            power_of_roots_shoup=cryptoContext.power_of_roots_shoup,
            power_of_roots=cryptoContext.power_of_roots,
        ).reshape(-1, cryptoContext.N)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, L0)


def mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    return _mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext)


def _mult_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.cur_limbs, monomial_degree, cryptoContext)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.cur_limbs, monomial_degree, cryptoContext)
    return cipher


# note: EvalBootstrap in ckksrns-fhe.cpp
# @utils.profile_pytorch_function
def eval_bootstrap(ciphertext, cryptoContext, bootstrap_constants, L0=None):
    if bootstrap_constants is None:
        raise ValueError("homo_bootstrap requires bootstrap_constants generated by easyfhe.bs.openfhe.generate(ctx, ...)")

    if L0 is None:
        L0 = ciphertext.cur_limbs
    logBsSlots = int(_constant_info(bootstrap_constants, "log_bs_slots"))
    level_budgets = list(_constant_info(bootstrap_constants, "level_budget"))
    M = cryptoContext.M
    N = cryptoContext.N
    slots = 1 << logBsSlots
    rescaleTech = cryptoContext.rescaleTech

    deg = _constant_scalar(bootstrap_constants, "degree")
    correctionFactor = _constant_scalar(bootstrap_constants, "correction_factor")
    correction = _constant_scalar(bootstrap_constants, "correction")

    if deg > correctionFactor:
        print(
            "Warning: Degree [",
            deg,
            "] must be less than or equal to the correction factor[",
            correctionFactor,
            "].",
        )

    pre = _constant_scalar(bootstrap_constants, "pre")
    scalar = _constant_scalar(bootstrap_constants, "post_scalar")

    # -------------------
    # raising the modulus
    # -------------------
    # In FLEXIBLEAUTO, raising the ciphertext to a larger number
    # of towers is a bit more complex, because we need to adjust
    # it's scaling factor to the one that corresponds to the level
    # it's being raised to.
    # Increasing the modulus

    tmp = ciphertext
    tmp = _rescale(tmp, tmp.noise_deg - 1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)

    # We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    # CKKS bootstrapping faster.
    raised = mod_raise(tmp, L0, cryptoContext)

    constantEvalMult = _constant_scalar(bootstrap_constants, "constant_eval_mult")

    raised = homo.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    ctxtDec = None  # Initialize decrypted ciphertext
    isLTBootstrap = level_budgets[0] == 1 and level_budgets[1] == 1

    if slots == M // 4:  # FULLY PACKED CASE
        # need to call internal modular reduction so it also works for FLEXIBLEAUTO
        raised = _rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(raised, cryptoContext, bootstrap_constants)

        conj = _homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEncI = homo.homo_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo.homo_add(ctxtEnc, conj, cryptoContext)
        ctxtEncI = mult_by_monomial_inplace(ctxtEncI, 3 * M // 4, cryptoContext)

        if ctxtEnc.noise_deg == 2: # noise_deg of ctxtEnc and ctxtEncI should be the same
            ctxtEnc = _rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = _rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        ctxtEnc = bootstrap_approx.eval_bootstrap_approx_mod(ctxtEnc, cryptoContext)
        ctxtEncI = bootstrap_approx.eval_bootstrap_approx_mod(ctxtEncI, cryptoContext)

        mult_by_monomial_inplace(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo.homo_add(ctxtEnc, ctxtEncI, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------

        # In the case of FLEXIBLEAUTO, we need one extra tower
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = _rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(ctxtEnc, cryptoContext, bootstrap_constants)

    else:  # SPARSELY PACKED CASE
        # -------------------
        # Running PartialSum
        # -------------------

        for step in range(int(math.log2(N // (2 * slots)))):
            temp = homo.homo_rotate(raised, (1 << step) * slots, cryptoContext)
            raised = homo.homo_add(raised, temp, cryptoContext)

        raised = raised.cipher_like(raised.cv, slots=slots)

        # ---------------------
        # Running CoeffsToSlots
        # ---------------------
        raised = _rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(raised, cryptoContext, bootstrap_constants)

        conj = _homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEnc = homo.homo_add(ctxtEnc, conj, cryptoContext)
        if ctxtEnc.noise_deg ==2 :
            ctxtEnc = _rescale(ctxtEnc, 1, cryptoContext)

        ctxtEnc = bootstrap_approx.eval_bootstrap_approx_mod(ctxtEnc, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------
        # In the case of FLEXIBLEAUTO, we need one extra tower
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = _rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(ctxtEnc, cryptoContext, bootstrap_constants)


        ctxtDec_rot = homo.homo_rotate(ctxtDec, slots, cryptoContext)
        ctxtDec = homo.homo_add(ctxtDec, ctxtDec_rot, cryptoContext)
        ctxtDec = ctxtDec.cipher_like(ctxtDec.cv, slots=slots)
        ctxtDec = ctxtDec.cipher_like(ctxtDec.cv, slots=ciphertext.slots)

    # 64-bit only: scale back the message to its original scale.
    ctxtDec = homo.homo_mul_scalar_int(ctxtDec, _constant_scalar(bootstrap_constants, "cor_factor"), cryptoContext)

    return ctxtDec.cipher_like(ctxtDec.cv, slots=ciphertext.slots)


def homo_bootstrap(cipher, cryptoContext, bootstrap_constants, L0=None):
    return _homo_bootstrap(cipher, cryptoContext, bootstrap_constants, L0=L0)


def _homo_bootstrap(cipher, cryptoContext, bootstrap_constants, L0=None):
    result = eval_bootstrap(cipher, cryptoContext, bootstrap_constants, L0=L0)

    # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
    result = _rescale(result, result.noise_deg - 1, cryptoContext)

    return result

def homo_double_bootstrap(cipher, L0, logBsSlots, level_budgets, precision, cryptoContext, bootstrap_constants):
    initSizeQ = cipher.cur_limbs

    # Step 1: Get the input.
    powerOfTwoModulus = 1 << precision

    # Step 2: Scale up by powerOfTwoModulus, and extend the modulus to powerOfTwoModulus * q.
    # Note that we extend the modulus implicitly without any code calls because the value always stays 0.
    ctScaledUp = cipher.deep_copy()
    # We multiply by powerOfTwoModulus, and leave the last CRT value to be 0 (mod powerOfTwoModulus). #todp:??
    ctScaledUp = homo.homo_mul_scalar_int(ctScaledUp, powerOfTwoModulus, cryptoContext)


    # Step 3: Bootstrap the initial ciphertext.
    ctInitialBootstrap = eval_bootstrap(cipher, cryptoContext, bootstrap_constants, L0=L0)
    ctInitialBootstrap = _rescale(ctInitialBootstrap, ctInitialBootstrap.noise_deg - 1, cryptoContext)

    # Step 4: Scale up by powerOfTwoModulus.
    ctInitialBootstrap = homo.homo_mul_scalar_int(ctInitialBootstrap, powerOfTwoModulus, cryptoContext)

    # Step 5: Mod-down to powerOfTwoModulus * q
    # We mod down, and leave the last CRT value to be 0 because it's divisible by powerOfTwoModulus.
    ctBootstrappedScaledDown = ctInitialBootstrap.deep_copy()
    bootstrappingSizeQ = ctBootstrappedScaledDown.cur_limbs
    # If we start with more towers, than we obtain from bootstrapping, return the original ciphertext.
    if bootstrappingSizeQ <= initSizeQ:
        return cipher.deep_copy()
    ctBootstrappedScaledDown.cur_limbs = cipher.cur_limbs # note: hard adjust, drop limbs regardless of the rescaleTech


    # Step 6 and 7: Calculate the bootstrapping error by subtracting the original ciphertext from the bootstrapped ciphertext. Mod down to q is done implicitly.
    ctBootstrappingError = homo.homo_sub(ctBootstrappedScaledDown, ctScaledUp, cryptoContext)

    # Step 8: Bootstrap the error.
    ctBootstrappingError = eval_bootstrap(ctBootstrappingError, cryptoContext, bootstrap_constants, L0=L0)
    ctBootstrappingError = _rescale(ctBootstrappingError, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

    # Step 9: Subtract the bootstrapped error from the initial bootstrap to get even lower error.
    finalCiphertext = homo.homo_sub(ctInitialBootstrap, ctBootstrappingError, cryptoContext)

    # Step 10: Scale back down by powerOfTwoModulus to get the original message.
    finalCiphertext = homo.homo_mul_scalar_double(finalCiphertext, 1.0 / powerOfTwoModulus, cryptoContext)

    # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
    finalCiphertext = _rescale(finalCiphertext, finalCiphertext.noise_deg - 1, cryptoContext)

    return finalCiphertext
