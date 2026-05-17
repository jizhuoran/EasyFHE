import easyfhe.fhe as fhe

try:
    from .fhe_state import reduce_noise_to_one, rescale_one_level
except ImportError:
    from fhe_state import reduce_noise_to_one, rescale_one_level

__all__ = [
    "aespa_add_shortcut",
    "aespa_nonlinear",
    "conv3x3",
    "initial_conv3x3",
    "pointwise_conv",
]


def _read_kernel_groups(prefixes, cipher, scale, cryptoContext, weights, is_ext=False, cache=True):
    level = cryptoContext.L - cipher.cur_limbs
    names = [
        f"{prefix}-k{k + 1}"
        for prefix in prefixes
        for k in range(9)
    ]
    return weights.plaintext_batch(names, level, cipher.slots, cryptoContext, scale, is_ext=is_ext, cache=cache)


def _conv3x3_offsets(img_width, padding):
    return [
        -img_width - padding,
        -img_width,
        -img_width + padding,
        -padding,
        0,
        padding,
        img_width - padding,
        img_width,
        img_width + padding,
    ]


def _validate_conv3x3_prefixes(prefixes):
    if not prefixes:
        raise ValueError("conv3x3 requires at least one kernel prefix")


def _conv3x3(input, prefixes, img_width, padding, rot_offset, scale, cryptoContext, weights):
    _validate_conv3x3_prefixes(prefixes)
    input = reduce_noise_to_one(input, cryptoContext)
    plaintexts = _read_kernel_groups(
        tuple(reversed(prefixes)),
        input,
        scale,
        cryptoContext,
        weights,
    )
    result = fhe.hoisted_mac_sum(
        input,
        _conv3x3_offsets(img_width, padding),
        plaintexts,
        rot_offset,
        len(prefixes),
        cryptoContext,
        strategy="normal",
    )
    return fhe.homo_rotate(result, rot_offset, cryptoContext)


def _initial_conv3x3(input, prefixes, img_width, padding, rot_offset, scale, cryptoContext, weights):
    _validate_conv3x3_prefixes(prefixes)
    input = reduce_noise_to_one(input, cryptoContext)
    rotations = fhe.fast_rotate_batch(input, _conv3x3_offsets(img_width, padding), cryptoContext)
    partial_sums = fhe.fused_grouped_pairwise_mac(
        rotations,
        _read_kernel_groups(tuple(reversed(prefixes)), rotations, scale, cryptoContext, weights),
        len(prefixes),
        cryptoContext,
    )
    partial_sums = [
        _initial_conv_postprocess(partial_sum, cryptoContext, weights)
        for partial_sum in partial_sums
    ]
    result = fhe.giant_rotate_sum(partial_sums, rot_offset, cryptoContext, strategy="normal")
    return fhe.homo_rotate(result, rot_offset, cryptoContext)


def _initial_conv_postprocess(partial_sum, cryptoContext, weights):
    partial_sum = rescale_one_level(partial_sum, cryptoContext)
    sum_rot = fhe.homo_rotate(partial_sum, 1024, cryptoContext)
    partial_sum = fhe.homo_add(partial_sum, sum_rot, cryptoContext)
    partial_sum = fhe.homo_add(partial_sum, fhe.homo_rotate(sum_rot, 1024, cryptoContext), cryptoContext)
    return fhe.homo_mul_pt(
        partial_sum,
        weights.plaintext(
            f"mask_from_to_0_1024_{partial_sum.slots}",
            cryptoContext.L - partial_sum.cur_limbs,
            partial_sum.slots,
            cryptoContext,
        ),
        cryptoContext,
    )


def initial_conv3x3(input, kernel_prefixes, img_width, padding, rot_offset, scale, cryptoContext, weights):
    return _initial_conv3x3(
        input,
        kernel_prefixes,
        img_width,
        padding,
        rot_offset,
        scale,
        cryptoContext,
        weights,
    )


def conv3x3(input, kernel_prefixes, img_width, padding, rot_offset, scale, cryptoContext, weights):
    return _conv3x3(
        input,
        kernel_prefixes,
        img_width,
        padding,
        rot_offset,
        scale,
        cryptoContext,
        weights,
    )


def pointwise_conv(input, kernel_keys, bias_key, rot_offset, scale, cryptoContext, weights):
    if not kernel_keys:
        raise ValueError("pointwise_conv requires at least one kernel key")

    input = reduce_noise_to_one(input, cryptoContext)

    for idx, kernel_key in enumerate(kernel_keys):
        encoded = weights.plaintext_for_cipher(kernel_key, input, cryptoContext, scale)
        partial_sum = fhe.homo_mul_pt(input, encoded, cryptoContext)
        finalsum = partial_sum.deep_copy() if idx == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)

    finalsum = rescale_one_level(finalsum, cryptoContext)
    bias = weights.plaintext_for_cipher(bias_key, finalsum, cryptoContext, scale)
    return fhe.homo_add_pt(finalsum, bias, cryptoContext)


def aespa_nonlinear(x, prefix, cryptoContext, weights, scale=1):
    x = reduce_noise_to_one(x, cryptoContext)
    n1 = weights.plaintext_for_cipher(f"{prefix}-n1", x, cryptoContext, scale)
    shifted = fhe.homo_add_pt(x, n1, cryptoContext)
    squared = fhe.homo_square(shifted, cryptoContext)
    squared = rescale_one_level(squared, cryptoContext)
    n2 = weights.plaintext_for_cipher(f"{prefix}-n2", squared, cryptoContext, scale)
    return fhe.homo_add_pt(squared, n2, cryptoContext)


def aespa_add_shortcut(conv_out, shortcut, prefix, cryptoContext, weights, scale=1):
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        shortcut = fhe.align_to(
            shortcut,
            fhe.CipherState(shortcut.cur_limbs - (shortcut.cur_limbs - conv_out.cur_limbs), shortcut.noise_deg),
            cryptoContext,
        )
    a2 = weights.plaintext_for_cipher(f"{prefix}-A2", shortcut, cryptoContext, scale)
    return fhe.homo_add(conv_out, fhe.homo_mul_pt(shortcut, a2, cryptoContext), cryptoContext)
