import math

import easyfhe.fhe as fhe

__all__ = [
    "aespa_add_shortcut",
    "aespa_nonlinear",
    "conv",
    "conv_initial",
    "convbn_dx",
    "downsample1024to256",
    "downsample256to64",
    "repeat",
    "rotsum",
    "rotsum_padded",
]


def _pairwise_mac(ctxs, ptxs, cryptoContext):
    if len(ctxs) != len(ptxs) or len(ctxs) == 0:
        raise ValueError(f"ctxs and ptxs must have the same non-zero length, but got {len(ctxs)} and {len(ptxs)}")

    partial_sum = fhe.homo_mul_pt(ctxs[0], ptxs[0], cryptoContext)
    for ctx, ptx in zip(ctxs[1:], ptxs[1:]):
        partial_sum = fhe.homo_add(partial_sum, fhe.homo_mul_pt(ctx, ptx, cryptoContext), cryptoContext)
    return partial_sum


def _rescale_one_level(cipher, cryptoContext):
    return fhe.align_to(cipher, fhe.CipherState(cipher.cur_limbs - 1, cipher.noise_deg - 1), cryptoContext)


def _reduce_noise_once_if_needed(cipher, cryptoContext):
    if cipher.noise_deg > 1:
        return _rescale_one_level(cipher, cryptoContext)
    return cipher


def _read_kernel_rows(prefix, cipher, scale, cryptoContext, weights):
    level = cryptoContext.L - cipher.cur_limbs
    return [
        weights.encode(f"{prefix}-k{k + 1}", level, cipher.slots, cryptoContext, scale)
        for k in range(9)
    ]


def _rot_input(input, img_width, padding, cryptoContext):
    digits = fhe.modup_to_ext(input.cipher_like([input.cv[1]]), cryptoContext)
    digits_neg_padding = fhe.eval_fast_rotate(digits, input, -padding, True, True, cryptoContext)
    digits_padding = fhe.eval_fast_rotate(digits, input, padding, True, True, cryptoContext)
    digits_neg_img_width = fhe.eval_fast_rotate(digits, input, -img_width, True, True, cryptoContext)
    digits_img_width = fhe.eval_fast_rotate(digits, input, img_width, True, True, cryptoContext)

    return [
        fhe.homo_rotate(digits_neg_padding, -img_width, cryptoContext),
        digits_neg_img_width,
        fhe.homo_rotate(digits_padding, -img_width, cryptoContext),
        digits_neg_padding,
        input,
        digits_padding,
        fhe.homo_rotate(digits_neg_padding, img_width, cryptoContext),
        digits_img_width,
        fhe.homo_rotate(digits_padding, img_width, cryptoContext),
    ]


def _conv3x3(input, prefixes, img_width, padding, rot_offset, scale, cryptoContext, weights, postprocess=None):
    input = _reduce_noise_once_if_needed(input, cryptoContext)
    rotations = _rot_input(input, img_width, padding, cryptoContext)

    for idx, prefix in enumerate(prefixes):
        partial_sum = _pairwise_mac(rotations, _read_kernel_rows(prefix, input, scale, cryptoContext, weights), cryptoContext)
        if postprocess is not None:
            partial_sum = postprocess(partial_sum)
        finalsum = partial_sum.deep_copy() if idx == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)
    return finalsum


def _initial_conv_postprocess(partial_sum, cryptoContext, weights):
    partial_sum = _rescale_one_level(partial_sum, cryptoContext)
    sum_rot = fhe.homo_rotate(partial_sum, 1024, cryptoContext)
    partial_sum = fhe.homo_add(partial_sum, sum_rot, cryptoContext)
    partial_sum = fhe.homo_add(partial_sum, fhe.homo_rotate(sum_rot, 1024, cryptoContext), cryptoContext)
    return fhe.homo_mul_pt(
        partial_sum,
        weights.encode(
            f"mask_from_to_0_1024_{partial_sum.slots}",
            cryptoContext.L - partial_sum.cur_limbs,
            partial_sum.slots,
            cryptoContext,
        ),
        cryptoContext,
    )


def conv_initial(input, img_width, padding, num_channel, scale, cryptoContext, weights):
    return _conv3x3(
        input,
        [f"conv1bn1-ch{j}" for j in range(num_channel)],
        img_width,
        padding,
        1024,
        scale,
        cryptoContext,
        weights,
        lambda partial_sum: _initial_conv_postprocess(partial_sum, cryptoContext, weights),
    )


def conv(input, img_width, padding, num_channel, rot_offset, layer, n, channel_offset, scale, cryptoContext, weights):
    return _conv3x3(
        input,
        [f"layer{layer}-conv{n}bn{n}-ch{j + channel_offset}" for j in range(num_channel)],
        img_width,
        padding,
        rot_offset,
        scale,
        cryptoContext,
        weights,
    )


def convbn_dx(input, num_channel, rot_offset, layer, n, channel_offset, biasoff, scale, cryptoContext, weights):
    input = _reduce_noise_once_if_needed(input, cryptoContext)

    for j in range(num_channel):
        encoded = weights.encode_for_cipher(
            f"layer{layer}dx-conv{n}bn{n}-ch{j + channel_offset}-k1",
            input,
            cryptoContext,
            scale,
        )
        partial_sum = fhe.homo_mul_pt(input, encoded, cryptoContext)
        finalsum = partial_sum.deep_copy() if j == 0 else fhe.homo_add(finalsum, partial_sum, cryptoContext)
        finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)

    finalsum = _rescale_one_level(finalsum, cryptoContext)
    bias = weights.encode_for_cipher(f"layer{layer}dx-conv{n}bn{n}-bias{biasoff}", finalsum, cryptoContext, scale)
    return fhe.homo_add_pt(finalsum, bias, cryptoContext)


def aespa_nonlinear(x, prefix, cryptoContext, weights, scale=1):
    x = _reduce_noise_once_if_needed(x, cryptoContext)
    n1 = weights.encode_for_cipher(f"{prefix}-n1", x, cryptoContext, scale)
    shifted = fhe.homo_add_pt(x, n1, cryptoContext)
    squared = fhe.homo_square(shifted, cryptoContext)
    squared = _rescale_one_level(squared, cryptoContext)
    n2 = weights.encode_for_cipher(f"{prefix}-n2", squared, cryptoContext, scale)
    return fhe.homo_add_pt(squared, n2, cryptoContext)


def aespa_add_shortcut(conv_out, shortcut, prefix, cryptoContext, weights, scale=1):
    if cryptoContext.rescaleTech == "FIXEDMANUAL":
        shortcut = fhe.align_to(
            shortcut,
            fhe.CipherState(shortcut.cur_limbs - (shortcut.cur_limbs - conv_out.cur_limbs), shortcut.noise_deg),
            cryptoContext,
        )
    a2 = weights.encode_for_cipher(f"{prefix}-A2", shortcut, cryptoContext, scale)
    return fhe.homo_add(conv_out, fhe.homo_mul_pt(shortcut, a2, cryptoContext), cryptoContext)


def _merge_fullpack(c1, c2, cryptoContext, weights):
    old_slots = c1.slots
    c1 = fhe.slot_resize(c1, c1.slots * 2, cryptoContext)
    c2 = fhe.slot_resize(c2, c2.slots * 2, cryptoContext)
    return fhe.homo_add(
        fhe.homo_mul_pt(
            c1,
            weights.encode(f"mask_first_n_{old_slots}_{c1.slots}", cryptoContext.L - c1.cur_limbs, c1.slots, cryptoContext),
            cryptoContext,
        ),
        fhe.homo_mul_pt(
            c2,
            weights.encode(f"mask_scecond_n_{old_slots}_{c2.slots}", cryptoContext.L - c2.cur_limbs, c2.slots, cryptoContext),
            cryptoContext,
        ),
        cryptoContext,
    )


def _double_rotate(cipher, cryptoContext):
    return fhe.homo_rotate(fhe.homo_rotate(cipher, 1, cryptoContext), 1, cryptoContext)


def _masked_reduce(cipher, mask_n, rotated, cryptoContext, weights):
    cipher = fhe.homo_mul_pt(
        fhe.homo_add(cipher, rotated, cryptoContext),
        weights.encode(f"gen_mask_{mask_n}_{cipher.slots}", cryptoContext.L - cipher.cur_limbs, cipher.slots, cryptoContext),
        cryptoContext,
    )
    return _rescale_one_level(cipher, cryptoContext)


def _spatial_reduce(fullpack, cryptoContext, weights, include_gen8, initial_rescale):
    if initial_rescale == "always":
        fullpack = _rescale_one_level(fullpack, cryptoContext)
    else:
        fullpack = _reduce_noise_once_if_needed(fullpack, cryptoContext)
    fullpack = _masked_reduce(fullpack, 2, fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext, weights)
    fullpack = _masked_reduce(fullpack, 4, _double_rotate(fullpack, cryptoContext), cryptoContext, weights)
    if include_gen8:
        fullpack = _masked_reduce(fullpack, 8, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext, weights)
        return fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 8, cryptoContext), cryptoContext)
    return fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext)


def _pack_rows(fullpack, row_mask_prefix, row_width, spatial_size, row_count, row_rotate, cryptoContext, weights):
    rows = None
    for i in range(row_count):
        masked = fhe.homo_mul_pt(
            fullpack,
            weights.encode(
                f"{row_mask_prefix}_{row_width}_{spatial_size}_{i}_{fullpack.slots}",
                cryptoContext.L - fullpack.cur_limbs,
                fullpack.slots,
                cryptoContext,
            ),
            cryptoContext,
        )
        rows = masked if rows is None else fhe.homo_add(rows, masked, cryptoContext)
        if i < row_count - 1:
            fullpack = fhe.homo_rotate(fullpack, row_rotate, cryptoContext)
    return _rescale_one_level(rows, cryptoContext)


def _pack_channels(rows, num_channel, spatial_size, out_spatial_size, cryptoContext, weights):
    channels = None
    for i in range(num_channel * 2):
        masked = fhe.homo_mul_pt(
            rows,
            weights.encode(
                f"mask_channel_{i}_{num_channel}_{spatial_size}",
                cryptoContext.L - rows.cur_limbs,
                rows.slots,
                cryptoContext,
            ),
            cryptoContext,
        )
        channels = masked if channels is None else fhe.homo_add(channels, masked, cryptoContext)
        channels = fhe.homo_rotate(channels, -(spatial_size - out_spatial_size), cryptoContext)
    return fhe.homo_rotate(channels, num_channel * 2 * (spatial_size - out_spatial_size), cryptoContext)


def _fold_quarters(cipher, cryptoContext):
    quarter = cipher.slots // 4
    cipher = fhe.homo_add(cipher, fhe.homo_rotate(cipher, -quarter, cryptoContext), cryptoContext)
    cipher = fhe.homo_add(cipher, fhe.homo_rotate(fhe.homo_rotate(cipher, -quarter, cryptoContext), -quarter, cryptoContext), cryptoContext)
    return cipher


def _downsample_spatial(c1, c2, num_channel, cryptoContext, weights, spec):
    fullpack = _merge_fullpack(c1, c2, cryptoContext, weights)
    fullpack = _spatial_reduce(
        fullpack,
        cryptoContext,
        weights,
        include_gen8=spec["include_gen8"],
        initial_rescale=spec["initial_rescale"],
    )
    rows = _pack_rows(
        fullpack,
        spec["row_mask_prefix"],
        spec["row_width"],
        spec["spatial_size"],
        spec["row_count"],
        spec["row_rotate"],
        cryptoContext,
        weights,
    )
    channels = _pack_channels(rows, num_channel, spec["spatial_size"], spec["out_spatial_size"], cryptoContext, weights)
    if spec["rescale_before_fold"]:
        channels = _rescale_one_level(channels, cryptoContext)
    channels = _fold_quarters(channels, cryptoContext)
    if spec["rescale_after_fold"]:
        channels = _rescale_one_level(channels, cryptoContext)
    return fhe.slot_resize(channels, channels.slots // 4, cryptoContext)


def downsample1024to256(c1, c2, num_channel, num_cipher, cryptoContext, weights):
    assert num_cipher == 1
    return _downsample_spatial(
        c1,
        c2,
        num_channel,
        cryptoContext,
        weights,
        {
            "spatial_size": 1024,
            "out_spatial_size": 256,
            "row_mask_prefix": "mask_first_n_mod",
            "row_width": 16,
            "row_count": 16,
            "row_rotate": 48,
            "include_gen8": True,
            "initial_rescale": "if_needed",
            "rescale_before_fold": True,
            "rescale_after_fold": False,
        },
    )


def downsample256to64(c1, c2, num_channel, cryptoContext, weights):
    return _downsample_spatial(
        c1,
        c2,
        num_channel,
        cryptoContext,
        weights,
        {
            "spatial_size": 256,
            "out_spatial_size": 64,
            "row_mask_prefix": "mask_first_n_mod2",
            "row_width": 8,
            "row_count": 32,
            "row_rotate": 24,
            "include_gen8": False,
            "initial_rescale": "always",
            "rescale_before_fold": False,
            "rescale_after_fold": True,
        },
    )


def rotsum(input, slots, cryptoContext):
    result = input.deep_copy()
    for i in range(int(math.log2(slots))):
        result = fhe.homo_add(result, fhe.homo_rotate(result, 2 ** i, cryptoContext), cryptoContext)
    return result


def rotsum_padded(input, slots, num_channel, cryptoContext):
    result = input.deep_copy()
    for i in range(int(math.log2(num_channel))):
        result = fhe.homo_add(result, fhe.homo_rotate(result, slots * (2 ** i), cryptoContext), cryptoContext)
    return result


def repeat(input, slots, cryptoContext):
    return fhe.homo_rotate(rotsum(input, slots, cryptoContext), -slots + 1, cryptoContext)
