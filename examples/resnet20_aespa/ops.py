import easyfhe.fhe as fhe

try:
    from .fhe_state import reduce_noise_to_one, rescale_one_level
except ImportError:
    from fhe_state import reduce_noise_to_one, rescale_one_level

__all__ = [
    "aespa_add_shortcut",
    "aespa_nonlinear",
    "conv3x3",
    "conv3x3_sx",
    "initial_conv3x3",
    "pointwise_conv",
    "pointwise_conv_sx",
]


def _read_kernel_group(name, cipher, scale, cryptoContext, weights, is_ext=False, cache=True):
    level = cryptoContext.L - cipher.state.cur_limbs
    return weights.plaintext(
        name,
        level,
        cipher.slots,
        cryptoContext,
        _resolve_scalar(scale, weights),
        is_ext=is_ext,
        cache=cache,
    )


def _resolve_scalar(value, weights):
    if isinstance(value, str):
        return weights._scalar_value(value)
    return value


def _cipher_batch_items(cipher):
    return tuple(
        cipher.cipher_like(
            [component[index] for component in cipher.cv],
            batch_size=1,
        )
        for index in range(int(cipher.batch_size))
    )


def _trace_op_state(cryptoContext, op, cipher, **extra):
    trace = getattr(cryptoContext, "aespa_state_trace", None)
    if trace is None:
        return
    record = {
        "op": op,
        "cur_limbs": int(cipher.state.cur_limbs),
        "noise_deg": int(cipher.state.noise_deg),
        "slots": int(cipher.slots),
        "is_ext": bool(cipher.is_ext),
    }
    record.update(extra)
    trace.append(record)


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


def _conv3x3_channel_count(plaintexts):
    if plaintexts.batch_size % 9 != 0:
        raise ValueError(f"conv3x3 kernel batch size must be a multiple of 9, got {plaintexts.batch_size}")
    return plaintexts.batch_size // 9


def _conv3x3(input, kernel_group, img_width, padding, rot_offset, scale, cryptoContext, weights):
    _trace_op_state(
        cryptoContext,
        "conv3x3",
        input,
        kernel_group=kernel_group,
        img_width=int(img_width),
        rot_offset=int(rot_offset),
    )
    input = reduce_noise_to_one(input, cryptoContext)
    plaintexts = _read_kernel_group(
        kernel_group,
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
        _conv3x3_channel_count(plaintexts),
        cryptoContext,
        strategy="normal",
    )
    return fhe.homo_rotate(result, rot_offset, cryptoContext)


def _conv3x3_sx(input, kernel_group, img_width, padding, copy_per_cipher, channels, rot_offset, scale, cryptoContext, weights):
    _trace_op_state(
        cryptoContext,
        "conv3x3_sx",
        input,
        kernel_group=kernel_group,
        img_width=int(img_width),
        rot_offset=int(rot_offset),
    )
    input = reduce_noise_to_one(input, cryptoContext)
    loop_size = int(channels) // int(copy_per_cipher)
    plaintexts = _read_kernel_group(
        kernel_group,
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
        loop_size,
        cryptoContext,
        strategy="normal",
    )
    return fhe.homo_rotate(result, loop_size * rot_offset, cryptoContext)


def _initial_conv3x3(input, kernel_group, img_width, padding, rot_offset, scale, cryptoContext, weights):
    _trace_op_state(
        cryptoContext,
        "initial_conv3x3",
        input,
        kernel_group=kernel_group,
        img_width=int(img_width),
        rot_offset=int(rot_offset),
    )
    input = reduce_noise_to_one(input, cryptoContext)
    rotations = fhe.fast_rotate(input, _conv3x3_offsets(img_width, padding), cryptoContext)
    plaintexts = _read_kernel_group(kernel_group, rotations, scale, cryptoContext, weights)
    partial_sums = fhe.grouped_pairwise_mac(
        rotations,
        plaintexts,
        _conv3x3_channel_count(plaintexts),
        cryptoContext,
    )
    partial_sums = [
        _initial_conv_postprocess(partial_sum, cryptoContext, weights)
        for partial_sum in _cipher_batch_items(partial_sums)
    ]
    result = fhe.giant_rotate_sum(partial_sums, rot_offset, cryptoContext, strategy="normal")
    return fhe.homo_rotate(result, rot_offset, cryptoContext)


def _initial_conv_postprocess(partial_sum, cryptoContext, weights):
    partial_sum = rescale_one_level(partial_sum, cryptoContext)
    base = partial_sum
    sum_rot = fhe.homo_rotate(partial_sum, 1024, cryptoContext)
    partial_sum = fhe.homo_rotate(sum_rot, 1024, cryptoContext, addend=sum_rot)
    partial_sum = fhe.homo_add(base, partial_sum, cryptoContext)
    return fhe.homo_mul_pt(
        partial_sum,
        weights.plaintext(
            f"mask_from_to_0_1024_{partial_sum.slots}",
            cryptoContext.L - partial_sum.state.cur_limbs,
            partial_sum.slots,
            cryptoContext,
        ),
        cryptoContext,
    )


def initial_conv3x3(input, kernel_group, img_width, padding, rot_offset, scale, cryptoContext, weights):
    return _initial_conv3x3(
        input,
        kernel_group,
        img_width,
        padding,
        rot_offset,
        scale,
        cryptoContext,
        weights,
    )


def conv3x3(input, kernel_group, img_width, padding, rot_offset, scale, cryptoContext, weights):
    return _conv3x3(
        input,
        kernel_group,
        img_width,
        padding,
        rot_offset,
        scale,
        cryptoContext,
        weights,
    )


def conv3x3_sx(input, kernel_group, img_width, padding, copy_per_cipher, channels, rot_offset, scale, cryptoContext, weights):
    return _conv3x3_sx(
        input,
        kernel_group,
        img_width,
        padding,
        copy_per_cipher,
        channels,
        rot_offset,
        scale,
        cryptoContext,
        weights,
    )


def pointwise_conv(input, kernel_group, bias_key, rot_offset, scale, cryptoContext, weights):
    _trace_op_state(
        cryptoContext,
        "pointwise_conv",
        input,
        kernel_group=kernel_group,
        bias_key=bias_key,
        rot_offset=int(rot_offset),
    )
    input = reduce_noise_to_one(input, cryptoContext)
    plaintexts = _read_kernel_group(kernel_group, input, scale, cryptoContext, weights)
    input_batch = input.cipher_like(input.cv, batch_size=1)
    partial_sums = fhe.grouped_pairwise_mac(input_batch, plaintexts, plaintexts.batch_size, cryptoContext)
    finalsum = fhe.giant_rotate_sum(partial_sums, rot_offset, cryptoContext, strategy="normal")
    finalsum = fhe.homo_rotate(finalsum, rot_offset, cryptoContext)

    finalsum = rescale_one_level(finalsum, cryptoContext)
    bias = weights.plaintext(
        bias_key,
        cryptoContext.L - finalsum.state.cur_limbs,
        finalsum.slots,
        cryptoContext,
        _resolve_scalar(scale, weights),
    )
    return fhe.homo_add_pt(finalsum, bias, cryptoContext)


def pointwise_conv_sx(input, kernel_group, bias_key, copy_per_cipher, channels, rot_offset, scale, cryptoContext, weights):
    _trace_op_state(
        cryptoContext,
        "pointwise_conv_sx",
        input,
        kernel_group=kernel_group,
        bias_key=bias_key,
        rot_offset=int(rot_offset),
    )
    input = reduce_noise_to_one(input, cryptoContext)
    loop_size = int(channels) // int(copy_per_cipher)
    plaintexts = _read_kernel_group(
        kernel_group,
        input,
        scale,
        cryptoContext,
        weights,
    )
    input_batch = input.cipher_like(input.cv, batch_size=1)
    partial_sums = fhe.grouped_pairwise_mac(input_batch, plaintexts, loop_size, cryptoContext)
    finalsum = fhe.giant_rotate_sum(partial_sums, rot_offset, cryptoContext, strategy="normal")
    finalsum = fhe.homo_rotate(finalsum, loop_size * rot_offset, cryptoContext)
    finalsum = rescale_one_level(finalsum, cryptoContext)
    bias = weights.plaintext(
        bias_key,
        cryptoContext.L - finalsum.state.cur_limbs,
        finalsum.slots,
        cryptoContext,
        _resolve_scalar(scale, weights),
    )
    return fhe.homo_add_pt(finalsum, bias, cryptoContext)


def aespa_nonlinear(x, prefix, cryptoContext, weights, scale=1):
    x = reduce_noise_to_one(x, cryptoContext)
    n1 = weights.plaintext(
        f"{prefix}-n1",
        cryptoContext.L - x.state.cur_limbs,
        x.slots,
        cryptoContext,
        _resolve_scalar(scale, weights),
    )
    shifted = fhe.homo_add_pt(x, n1, cryptoContext)
    out_cur_limbs = shifted.state.cur_limbs - 1
    n2 = weights.plaintext(
        f"{prefix}-n2",
        cryptoContext.L - out_cur_limbs,
        shifted.slots,
        cryptoContext,
        _resolve_scalar(scale, weights),
    )
    return fhe.homo_mul_relin_rescale_add_pt(shifted, shifted, n2, cryptoContext)


def aespa_add_shortcut(conv_out, shortcut, prefix, cryptoContext, weights, scale=1):
    if cryptoContext.scale_mode == "fixed" and cryptoContext.rescale_policy == "manual":
        shortcut = fhe.align_to(
            shortcut,
            fhe.CipherState(shortcut.state.cur_limbs - (shortcut.state.cur_limbs - conv_out.state.cur_limbs), shortcut.state.noise_deg),
            cryptoContext,
        )
    a2 = weights.plaintext(
        f"{prefix}-A2",
        cryptoContext.L - shortcut.state.cur_limbs,
        shortcut.slots,
        cryptoContext,
        _resolve_scalar(scale, weights),
    )
    return fhe.homo_add(conv_out, fhe.homo_mul_pt(shortcut, a2, cryptoContext), cryptoContext)
