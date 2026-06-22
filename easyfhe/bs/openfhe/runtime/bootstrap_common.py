from easyfhe.fhe.ops import kernels as F
from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import rotation
from . import approx as bootstrap_approx


def coeffs_slots_conversion(ciphertext, transform_plan, constants, bootstrap_plan, cryptoContext):
    result = ciphertext
    strategy = bootstrap_plan.strategy
    hoist_strategy = {
        "normal_bsgs": rotation.HOIST_NORMAL,
        "normal_giant": rotation.HOIST_EXT_NORMAL,
    }.get(strategy, rotation.HOIST_EXT_DOUBLE_HOIST)
    is_ext = strategy != "normal_bsgs"

    for loop_pos, step in enumerate(transform_plan.steps):
        if loop_pos != 0:
            result = alignment.reduce_noise_to_one(result, cryptoContext)

        plaintext_batch = constants.plaintext(
            step.plaintext_name,
            cryptoContext.L - result.state.cur_limbs,
            step.plaintext_slots,
            cryptoContext,
            is_ext=is_ext,
            scaling_factor=_transform_plaintext_encode_scale(result.state, transform_plan, cryptoContext),
            cur_limbs=result.state.cur_limbs,
        )
        result = rotation.hoisted_mac_sum(
            result,
            step.input_offsets,
            plaintext_batch,
            step.giant_offset,
            step.baby_step,
            cryptoContext,
            strategy=hoist_strategy,
        )
    return result


def eval_coeffs_to_slots(ctxt, cryptoContext, bootstrap_constants, bootstrap_plan):
    return coeffs_slots_conversion(
        ctxt,
        bootstrap_plan.c2s_plan,
        bootstrap_constants,
        bootstrap_plan,
        cryptoContext,
    )


def eval_slots_to_coeffs(ctxt, cryptoContext, bootstrap_constants, bootstrap_plan):
    return coeffs_slots_conversion(
        ctxt,
        bootstrap_plan.s2c_plan,
        bootstrap_constants,
        bootstrap_plan,
        cryptoContext,
    )


def mod_raise(cipher, L0, cryptoContext):
    cv = [
        F.cv_mod_raise(
            cv,
            L0,
            cryptoContext,
        )
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, state=cipher.state.replace(cur_limbs=L0))


def mul_by_monomial_inplace(cipher, monomial_degree, cryptoContext):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.state.cur_limbs, monomial_degree, cryptoContext)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.state.cur_limbs, monomial_degree, cryptoContext)
    return cipher


def raise_ciphertext(ciphertext, cryptoContext, bootstrap_constants, L0):
    correction_scale = bootstrap_constants._scalar_value("correction_scale")
    result = alignment.reduce_noise_to_one(ciphertext, cryptoContext)

    source_scale = float(result.state.scaling_factor)
    target_scale = float(cryptoContext.scale_at(cryptoContext.L))
    correction_drop_count = int(getattr(cryptoContext, "first_mod_limb_count", 1))
    correction_divisor = float(
        cryptoContext.physical_rescale_divisor_for_limbs(
            result.state.cur_limbs,
            correction_drop_count,
        )
    )
    correction_encoding_scale = float(cryptoContext.scale_at(result.state.cur_limbs))
    adjustment_factor = (
        target_scale
        / source_scale
        * correction_divisor
        / source_scale
        * correction_scale
    )

    result = arithmetic.homo_mul_scalar_double(
        result,
        arithmetic._encode_double_for_scalar_op(
            adjustment_factor,
            result.state.cur_limbs,
            cryptoContext,
            scaling_factor=correction_encoding_scale,
        ),
        cryptoContext,
        scaling_factor=correction_encoding_scale,
    )
    result = alignment.rescale_one_level(result, cryptoContext)
    result = result.cipher_like(result.cv, state=result.state.replace(scaling_factor=target_scale))

    result = mod_raise(result, L0, cryptoContext)
    scalar = bootstrap_constants.encoded_scalars(
        "constant_eval_mult",
        result.state.cur_limbs,
        1,
        cryptoContext,
        mode="double",
        scaling_factor=result.state.scaling_factor,
    )[0]
    result = arithmetic.homo_mul_scalar_double(
        result,
        scalar,
        cryptoContext,
        scaling_factor=result.state.scaling_factor,
    )
    return alignment.reduce_noise_to_one(result, cryptoContext)


def scale_after_approx(ciphertext, cryptoContext, bootstrap_constants):
    scalar = bootstrap_constants.encoded_scalars(
        "post_scalar", ciphertext.state.cur_limbs, 0, cryptoContext, mode="int"
    )[0]
    result = arithmetic.homo_mul_scalar_int_inplace(
        ciphertext,
        scalar,
        cryptoContext,
    )
    if cryptoContext.scale_mode == "flexible":
        return result.cipher_like(result.cv, state=result.state.replace(scaling_factor=cryptoContext.scale_at(cryptoContext.L)))
    return result


def scale_to_original_message(ciphertext, cryptoContext, bootstrap_constants):
    scalar = bootstrap_constants.encoded_scalars(
        "cor_factor", ciphertext.state.cur_limbs, 0, cryptoContext, mode="int"
    )[0]
    return arithmetic.homo_mul_scalar_int_inplace(
        ciphertext,
        scalar,
        cryptoContext,
    )


def eval_mod_full(encoded, cryptoContext, bootstrap_constants, bootstrap_plan):
    conjugate = rotation.homo_rotate(encoded, 2 * cryptoContext.N - 1, cryptoContext)
    imag = arithmetic.homo_sub(encoded, conjugate, cryptoContext)
    real = arithmetic.homo_add(encoded, conjugate, cryptoContext)
    imag = mul_by_monomial_inplace(imag, 3 * cryptoContext.M // 4, cryptoContext)

    real = alignment.reduce_noise_to_one(real, cryptoContext)
    imag = alignment.reduce_noise_to_one(imag, cryptoContext)

    real = bootstrap_approx.eval_bootstrap_approx_mod(real, cryptoContext, bootstrap_constants, bootstrap_plan)
    imag = bootstrap_approx.eval_bootstrap_approx_mod(imag, cryptoContext, bootstrap_constants, bootstrap_plan)

    imag = mul_by_monomial_inplace(imag, cryptoContext.M // 4, cryptoContext)
    encoded = arithmetic.homo_add(real, imag, cryptoContext)
    encoded = scale_after_approx(encoded, cryptoContext, bootstrap_constants)
    return alignment.reduce_noise_to_one(encoded, cryptoContext)


def eval_mod_sparse(encoded, cryptoContext, bootstrap_constants, bootstrap_plan):
    encoded = rotation.homo_rotate_add(encoded, 2 * cryptoContext.N - 1, cryptoContext, addend=encoded)
    encoded = alignment.reduce_noise_to_one(encoded, cryptoContext)
    encoded = bootstrap_approx.eval_bootstrap_approx_mod(encoded, cryptoContext, bootstrap_constants, bootstrap_plan)
    encoded = scale_after_approx(encoded, cryptoContext, bootstrap_constants)
    return alignment.reduce_noise_to_one(encoded, cryptoContext)


def _transform_plaintext_encode_scale(state, transform_plan, cryptoContext):
    del transform_plan
    return state.scaling_factor
