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


def _pairwise_mac(ctxs, ptxs, cryptoContext):
    if hasattr(ctxs, "batch_size") and int(ctxs.batch_size) > 1:
        if not hasattr(ptxs, "batch_size"):
            raise TypeError("batched _pairwise_mac requires batched plaintexts")
        return fhe.fused_pairwise_mac(ctxs, ptxs, cryptoContext)

    if len(ctxs) != len(ptxs) or len(ctxs) == 0:
        raise ValueError(f"ctxs and ptxs must have the same non-zero length, but got {len(ctxs)} and {len(ptxs)}")

    partial_sum = fhe.homo_mul_pt(ctxs[0], ptxs[0], cryptoContext)
    for ctx, ptx in zip(ctxs[1:], ptxs[1:]):
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_mul_pt(ctx, ptx, cryptoContext), cryptoContext)
    return partial_sum


def _read_kernel_rows(prefix, cipher, scale, cryptoContext, weights):
    level = cryptoContext.L - cipher.cur_limbs
    names = [f"{prefix}-k{k + 1}" for k in range(9)]
    if hasattr(cipher, "batch_size") and int(cipher.batch_size) > 1:
        return weights.plaintext_batch(names, level, cipher.slots, cryptoContext, scale)
    return [
        weights.plaintext(name, level, cipher.slots, cryptoContext, scale)
        for name in names
    ]


def _read_kernel_groups(prefixes, cipher, scale, cryptoContext, weights):
    level = cryptoContext.L - cipher.cur_limbs
    names = [
        f"{prefix}-k{k + 1}"
        for prefix in prefixes
        for k in range(9)
    ]
    return weights.plaintext_batch(names, level, cipher.slots, cryptoContext, scale)


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


def _rot_input(input, img_width, padding, cryptoContext):
    return fhe.fast_rotate_batch(input, _conv3x3_offsets(img_width, padding), cryptoContext)


def _validate_conv3x3_prefixes(prefixes):
    if not prefixes:
        raise ValueError("conv3x3 requires at least one kernel prefix")


def _batch_item(cipher, index):
    if int(getattr(cipher, "batch_size", 1)) <= 1 and cipher.cv[0].dim() == 2:
        if int(index) != 0:
            raise IndexError(f"batch index {index} out of range")
        return cipher
    batch_size = int(getattr(cipher, "batch_size", cipher.cv[0].shape[0]))
    index = int(index)
    if index < 0 or index >= batch_size:
        raise IndexError(f"batch index {index} out of range for batch_size={batch_size}")
    return cipher.cipher_like([cv[index] for cv in cipher.cv], batch_size=1, cipher_id="assign")


def _conv3x3(input, prefixes, img_width, padding, rot_offset, scale, cryptoContext, weights):
    _validate_conv3x3_prefixes(prefixes)
    input = reduce_noise_to_one(input, cryptoContext)
    rotations = _rot_input(input, img_width, padding, cryptoContext)
    partial_sums = fhe.fused_grouped_pairwise_mac(
        rotations,
        _read_kernel_groups(prefixes, rotations, scale, cryptoContext, weights),
        len(prefixes),
        cryptoContext,
    )

    finalsum = None
    for idx in range(len(prefixes)):
        partial_sum = _batch_item(partial_sums, idx)
        finalsum = partial_sum.deep_copy() if idx == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)
    return finalsum


def _initial_conv3x3(input, prefixes, img_width, padding, rot_offset, scale, cryptoContext, weights):
    _validate_conv3x3_prefixes(prefixes)
    input = reduce_noise_to_one(input, cryptoContext)
    rotations = _rot_input(input, img_width, padding, cryptoContext)
    partial_sums = fhe.fused_grouped_pairwise_mac(
        rotations,
        _read_kernel_groups(prefixes, rotations, scale, cryptoContext, weights),
        len(prefixes),
        cryptoContext,
    )
    finalsum = None
    for idx in range(len(prefixes)):
        partial_sum = _batch_item(partial_sums, idx)
        partial_sum = _initial_conv_postprocess(partial_sum, cryptoContext, weights)
        finalsum = partial_sum.deep_copy() if idx == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)
    return finalsum


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
