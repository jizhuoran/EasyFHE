from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import rotation


def coeffs_slots_conversion(ciphertext, transform_plan, constants, bootstrap_plan, crypto_context):
    result = ciphertext
    strategy = bootstrap_plan.strategy
    hoist_strategy = {
        "normal_bsgs": "normal",
        "normal_giant": "ext_normal",
    }.get(strategy, "ext_double_hoist")
    is_ext = strategy != "normal_bsgs"

    for loop_pos, step in enumerate(transform_plan.steps):
        if loop_pos != 0:
            result = alignment.normalize_scale(result, crypto_context)

        plaintext_batch = constants.plaintext(
            step.plaintext_name,
            state=result.state,
            slots=step.plaintext_slots,
            context=crypto_context,
            is_ext=is_ext,
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
