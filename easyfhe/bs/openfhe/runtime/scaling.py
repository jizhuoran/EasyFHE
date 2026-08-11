from easyfhe.fhe.ops import arithmetic


def scale_after_approx(ciphertext, crypto_context, bootstrap_constants, *, raise_to_limbs):
    scalar = bootstrap_constants.encoded_scalars(
        "post_scalar", ciphertext.state.cur_limbs, 0, crypto_context, mode="int"
    )[0]
    result = arithmetic.homo_mul_scalar_int_inplace(
        ciphertext,
        scalar,
        crypto_context,
    )
    if crypto_context.scale_mode == "flexible":
        return result.cipher_like(
            result.cv,
            state=result.state.replace(scaling_factor=crypto_context.scale_at(raise_to_limbs)),
        )
    return result


def scale_to_original_message(ciphertext, crypto_context, bootstrap_constants):
    scalar = bootstrap_constants.encoded_scalars(
        "cor_factor", ciphertext.state.cur_limbs, 0, crypto_context, mode="int"
    )[0]
    return arithmetic.homo_mul_scalar_int_inplace(
        ciphertext,
        scalar,
        crypto_context,
    )
