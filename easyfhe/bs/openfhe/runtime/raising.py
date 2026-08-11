from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import kernels as F


def mod_raise(cipher, target_limbs, crypto_context):
    cv = [
        F.cv_mod_raise(component, target_limbs, crypto_context)
        for component in cipher.cv
    ]
    return cipher.cipher_like(cv, state=cipher.state.replace(cur_limbs=target_limbs))


def raise_ciphertext(ciphertext, crypto_context, bootstrap_constants, raise_to_limbs):
    """Run the u64 bootstrap raise path (one physical limb per rescale)."""
    correction_scale = bootstrap_constants._scalar_value("correction_scale")
    result = alignment.reduce_noise_to_one(ciphertext, crypto_context)

    source_scale = float(result.state.scaling_factor)
    raise_to_limbs = int(raise_to_limbs)
    target_scale = float(crypto_context.scale_at(raise_to_limbs))
    correction_divisor = float(
        crypto_context.physical_rescale_divisor_for_limbs(result.state.cur_limbs, 1)
    )
    correction_encoding_scale = float(crypto_context.scale_at(result.state.cur_limbs))
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
            crypto_context,
            scaling_factor=correction_encoding_scale,
        ),
        crypto_context,
        scaling_factor=correction_encoding_scale,
    )
    result = alignment.rescale_one_level(result, crypto_context)
    result = result.cipher_like(result.cv, state=result.state.replace(scaling_factor=target_scale))

    result = mod_raise(result, raise_to_limbs, crypto_context)
    scalar = bootstrap_constants.encoded_scalars(
        "constant_eval_mult",
        result.state.cur_limbs,
        1,
        crypto_context,
        mode="double",
        scaling_factor=result.state.scaling_factor,
    )[0]
    result = arithmetic.homo_mul_scalar_double(
        result,
        scalar,
        crypto_context,
        scaling_factor=result.state.scaling_factor,
    )
    return alignment.reduce_noise_to_one(result, crypto_context)
