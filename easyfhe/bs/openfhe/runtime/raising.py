from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import kernels as F
from easyfhe.fhe.constants import encode_scalar


def mod_raise(cipher, target_limbs, crypto_context):
    cv = [
        F.cv_mod_raise(component, target_limbs, crypto_context)
        for component in cipher.cv
    ]
    return cipher.cipher_like(cv, state=cipher.state.replace(cur_limbs=target_limbs))


def raise_ciphertext(ciphertext, crypto_context, bootstrap_constants, raise_to_limbs):
    """Run the u64 bootstrap raise path (one physical limb per rescale)."""
    correction_scale = bootstrap_constants._scalar_value("correction_scale")
    result = alignment.normalize_scale(ciphertext, crypto_context)

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

    result = arithmetic.homo_mul_scalar(
        result,
        encode_scalar(
            adjustment_factor,
            cur_limbs=result.state.cur_limbs,
            scale_degree=1,
            context=crypto_context,
            scaling_factor=correction_encoding_scale,
        ),
        crypto_context,
    )
    result = alignment.rescale(result, crypto_context)
    result = result.cipher_like(result.cv, state=result.state.replace(scaling_factor=target_scale))

    result = mod_raise(result, raise_to_limbs, crypto_context)
    scalar = bootstrap_constants.encoded_scalars(
        "constant_eval_mult",
        cur_limbs=result.state.cur_limbs,
        scale_degree=1,
        context=crypto_context,
        mode="scaled",
        scaling_factor=result.state.scaling_factor,
    )[0]
    result = arithmetic.homo_mul_scalar(
        result,
        scalar,
        crypto_context,
    )
    return alignment.normalize_scale(result, crypto_context)
