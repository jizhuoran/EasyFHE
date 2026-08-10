from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import rotation


def coeffs_slots_conversion(ciphertext, transform_plan, constants, bootstrap_plan, crypto_context):
    result = ciphertext
    strategy = bootstrap_plan.strategy
    hoist_strategy = {
        "normal_bsgs": rotation.HOIST_NORMAL,
        "normal_giant": rotation.HOIST_EXT_NORMAL,
    }.get(strategy, rotation.HOIST_EXT_DOUBLE_HOIST)
    is_ext = strategy != "normal_bsgs"

    for loop_pos, step in enumerate(transform_plan.steps):
        if loop_pos != 0:
            result = alignment.reduce_noise_to_one(result, crypto_context)

        plaintext_batch = constants.plaintext(
            step.plaintext_name,
            crypto_context.L - result.state.cur_limbs,
            step.plaintext_slots,
            crypto_context,
            is_ext=is_ext,
            scaling_factor=result.state.scaling_factor,
            cur_limbs=result.state.cur_limbs,
        )
        result = rotation.hoisted_mac_sum(
            result,
            step.input_offsets,
            plaintext_batch,
            step.giant_offset,
            step.baby_step,
            crypto_context,
            strategy=hoist_strategy,
        )
    return result


def eval_coeffs_to_slots(ciphertext, crypto_context, bootstrap_constants, bootstrap_plan):
    return coeffs_slots_conversion(
        ciphertext,
        bootstrap_plan.c2s_plan,
        bootstrap_constants,
        bootstrap_plan,
        crypto_context,
    )


def eval_slots_to_coeffs(ciphertext, crypto_context, bootstrap_constants, bootstrap_plan):
    return coeffs_slots_conversion(
        ciphertext,
        bootstrap_plan.s2c_plan,
        bootstrap_constants,
        bootstrap_plan,
        crypto_context,
    )
