from dataclasses import dataclass

import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe

try:
    from .layout import (
        broadcast_slot_sum,
        downsample1024to256,
        downsample256to64,
        sum_adjacent_slots,
        sum_channel_groups,
    )
    from .fhe_state import rescale_one_level
    from .ops import (
        aespa_add_shortcut,
        aespa_nonlinear,
        conv3x3,
        initial_conv3x3,
        pointwise_conv,
    )
    from .weight_pack import WeightPack
except ImportError:
    from layout import (
        broadcast_slot_sum,
        downsample1024to256,
        downsample256to64,
        sum_adjacent_slots,
        sum_channel_groups,
    )
    from fhe_state import rescale_one_level
    from ops import (
        aespa_add_shortcut,
        aespa_nonlinear,
        conv3x3,
        initial_conv3x3,
        pointwise_conv,
    )
    from weight_pack import WeightPack


@dataclass
class AespaRuntime:
    ctx: object
    weights: WeightPack
    config: object
    bootstrap_constants: dict[int, object]


@dataclass(frozen=True)
class _SameShapeBlockSpec:
    block_id: int
    img_width: int
    channels: int
    rot_offset: int
    bootstrap_l0: object = None


@dataclass(frozen=True)
class _DownsampleBlockSpec:
    block_id: int
    in_img_width: int
    in_channels: int
    out_img_width: int
    out_channels: int
    first_rot: int
    second_rot: int
    downsample_kind: str
    bootstrap_l0: object
    rescale_after_add: bool = False


def _convbn_weight_prefix(block_id, conv_id):
    return f"layer{block_id}-conv{conv_id}bn{conv_id}"


def _conv3x3_kernel_prefixes(block_id, conv_id, channels, channel_offset=0):
    prefix = _convbn_weight_prefix(block_id, conv_id)
    return [f"{prefix}-ch{channel + channel_offset}" for channel in range(channels)]


def _pointwise_kernel_keys(block_id, conv_id, channels, channel_offset=0):
    return [
        f"layer{block_id}dx-conv{conv_id}bn{conv_id}-ch{channel + channel_offset}-k1"
        for channel in range(channels)
    ]


def _pointwise_bias_key(block_id, conv_id, bias_offset):
    return f"layer{block_id}dx-conv{conv_id}bn{conv_id}-bias{bias_offset}"


def _conv_then_aespa_nonlinear(input, block_id, conv_id, img_width, channels, rot_offset, rt, scale=1):
    res = conv3x3(
        input,
        _conv3x3_kernel_prefixes(block_id, conv_id, channels),
        img_width,
        1,
        rot_offset,
        scale,
        rt.ctx,
        rt.weights,
    )
    res = rescale_one_level(res, rt.ctx)
    return aespa_nonlinear(res, _convbn_weight_prefix(block_id, conv_id), rt.ctx, rt.weights, scale)


def _downsample_spatial_pair(sx0, sx1, dx0, dx1, in_channels, downsample_kind, rt):
    if downsample_kind == "1024to256":
        return (
            downsample1024to256(sx0, sx1, in_channels, rt.ctx, rt.weights),
            downsample1024to256(dx0, dx1, in_channels, rt.ctx, rt.weights),
        )
    if downsample_kind == "256to64":
        return (
            downsample256to64(sx0, sx1, in_channels, rt.ctx, rt.weights),
            downsample256to64(dx0, dx1, in_channels, rt.ctx, rt.weights),
        )
    raise ValueError(f"Unsupported downsample kind: {downsample_kind}")


def _downsample_conv_pair(input, block_id, in_img_width, in_channels, first_rot, rt, scale):
    first_half = conv3x3(
        input,
        _conv3x3_kernel_prefixes(block_id, 1, in_channels),
        in_img_width,
        1,
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    second_half = conv3x3(
        input,
        _conv3x3_kernel_prefixes(block_id, 1, in_channels, channel_offset=in_channels),
        in_img_width,
        1,
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    return rescale_one_level(first_half, rt.ctx), rescale_one_level(second_half, rt.ctx)


def _projection_input_for_downsample(input, rt):
    if rt.ctx.rescaleTech != "FIXEDMANUAL":
        return input
    return fhe.align_to(input, fhe.CipherState(input.cur_limbs - 2, input.noise_deg), rt.ctx)


def _downsample_projection_pair(input, block_id, in_channels, first_rot, rt, scale):
    input = _projection_input_for_downsample(input, rt)
    first_half = pointwise_conv(
        input,
        _pointwise_kernel_keys(block_id, 1, in_channels),
        _pointwise_bias_key(block_id, 1, "1"),
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    second_half = pointwise_conv(
        input,
        _pointwise_kernel_keys(block_id, 1, in_channels, channel_offset=in_channels),
        _pointwise_bias_key(block_id, 1, "2"),
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    return first_half, second_half


def _same_shape_residual_block(
    input,
    spec,
    rt,
):
    scale = 1
    res = _conv_then_aespa_nonlinear(
        input,
        spec.block_id,
        1,
        spec.img_width,
        spec.channels,
        spec.rot_offset,
        rt,
        scale,
    )
    res = conv3x3(
        res,
        _conv3x3_kernel_prefixes(spec.block_id, 2, spec.channels),
        spec.img_width,
        1,
        spec.rot_offset,
        scale,
        rt.ctx,
        rt.weights,
    )
    res = aespa_add_shortcut(res, input, _convbn_weight_prefix(spec.block_id, 2), rt.ctx, rt.weights, scale)
    res = rescale_one_level(res, rt.ctx)

    if spec.bootstrap_l0 is not None:
        log_bs_slots = rt.config.log_bs_slots[0]
        res = bs.bootstrap(
            res,
            rt.ctx,
            rt.bootstrap_constants[int(log_bs_slots)],
            L0=spec.bootstrap_l0,
        )
    return aespa_nonlinear(res, _convbn_weight_prefix(spec.block_id, 2), rt.ctx, rt.weights, scale)


def _downsample_residual_block(
    input,
    spec,
    rt,
):
    scale_sx = 1
    scale_dx = 1

    sx0, sx1 = _downsample_conv_pair(
        input,
        spec.block_id,
        spec.in_img_width,
        spec.in_channels,
        spec.first_rot,
        rt,
        scale_sx,
    )
    dx0, dx1 = _downsample_projection_pair(input, spec.block_id, spec.in_channels, spec.first_rot, rt, scale_dx)
    sx, dx = _downsample_spatial_pair(sx0, sx1, dx0, dx1, spec.in_channels, spec.downsample_kind, rt)

    sx = rescale_one_level(sx, rt.ctx)
    sx = aespa_nonlinear(sx, _convbn_weight_prefix(spec.block_id, 1), rt.ctx, rt.weights, scale_sx)
    sx = conv3x3(
        sx,
        _conv3x3_kernel_prefixes(spec.block_id, 2, spec.out_channels),
        spec.out_img_width,
        1,
        spec.second_rot,
        scale_dx,
        rt.ctx,
        rt.weights,
    )
    res = fhe.homo_add(sx, dx, rt.ctx)
    if spec.rescale_after_add:
        res = rescale_one_level(res, rt.ctx)

    log_bs_slots = rt.config.log_bs_slots[0]
    res = bs.bootstrap(
        res,
        rt.ctx,
        rt.bootstrap_constants[int(log_bs_slots)],
        L0=spec.bootstrap_l0,
    )
    return aespa_nonlinear(res, _convbn_weight_prefix(spec.block_id, 2), rt.ctx, rt.weights, scale_dx)


def initial_layer(input, rt):
    res = initial_conv3x3(
        input,
        [f"conv1bn1-ch{channel}" for channel in range(16)],
        32,
        1,
        1024,
        1,
        rt.ctx,
        rt.weights,
    )
    res = rescale_one_level(res, rt.ctx)
    return aespa_nonlinear(res, "conv1bn1", rt.ctx, rt.weights)


def layer1(input, rt):
    res = _same_shape_residual_block(input, _SameShapeBlockSpec(1, 32, 16, -1024), rt)
    res = _same_shape_residual_block(
        res,
        _SameShapeBlockSpec(
            2,
            32,
            16,
            -1024,
            bootstrap_l0=rt.ctx.L - (rt.config.max_levels_remaining - 5),
        ),
        rt,
    )
    return _same_shape_residual_block(
        res,
        _SameShapeBlockSpec(3, 32, 16, -1024, bootstrap_l0=rt.ctx.L),
        rt,
    )


def layer2(input, rt):
    res = _downsample_residual_block(
        input,
        _DownsampleBlockSpec(
            block_id=4,
            in_img_width=32,
            in_channels=16,
            out_img_width=16,
            out_channels=32,
            first_rot=-1024,
            second_rot=-256,
            downsample_kind="1024to256",
            bootstrap_l0=rt.ctx.L - (rt.config.max_levels_remaining - 9),
        ),
        rt,
    )
    res = _same_shape_residual_block(res, _SameShapeBlockSpec(5, 16, 32, -256), rt)
    return _same_shape_residual_block(res, _SameShapeBlockSpec(6, 16, 32, -256, bootstrap_l0=rt.ctx.L - 1), rt)


def layer3(input, rt):
    res = _downsample_residual_block(
        input,
        _DownsampleBlockSpec(
            block_id=7,
            in_img_width=16,
            in_channels=32,
            out_img_width=8,
            out_channels=64,
            first_rot=-256,
            second_rot=-64,
            downsample_kind="256to64",
            bootstrap_l0=rt.ctx.L,
            rescale_after_add=True,
        ),
        rt,
    )
    res = _same_shape_residual_block(res, _SameShapeBlockSpec(8, 8, 64, -64), rt)
    return _same_shape_residual_block(res, _SameShapeBlockSpec(9, 8, 64, -64), rt)


def final_layer(input, rt):
    channels = 64
    spatial_size = 64
    fc_repeat = 16

    res = sum_adjacent_slots(input, spatial_size, rt.ctx)
    res = fhe.homo_mul_pt(
        res,
        rt.weights.plaintext(
            f"mask_mod_{spatial_size}_{1.0 / spatial_size}_{res.slots}",
            rt.ctx.L - res.cur_limbs,
            res.slots,
            rt.ctx,
        ),
        rt.ctx,
    )
    res = broadcast_slot_sum(res, fc_repeat, rt.ctx)
    res = rescale_one_level(res, rt.ctx)
    weight = rt.weights.plaintext_for_cipher(f"fc_{res.slots}", res, rt.ctx)
    res = fhe.homo_mul_pt(res, weight, rt.ctx)
    res = rescale_one_level(res, rt.ctx)
    res = sum_channel_groups(res, spatial_size, channels, rt.ctx)

    bias = rt.weights.plaintext_for_cipher(f"bias_{res.slots}", res, rt.ctx)
    return fhe.homo_add_pt(res, bias, rt.ctx)


def infer_one(image_vector, rt):
    in_ct = rt.ctx.encrypt(image_vector, rt.ctx.device, 1, 19, 16 * 32 * 32)
    first_layer = initial_layer(in_ct, rt)
    res_layer1 = layer1(first_layer, rt)
    res_layer2 = layer2(res_layer1, rt)
    res_layer3 = layer3(res_layer2, rt)
    return final_layer(res_layer3, rt)


__all__ = [
    "AespaRuntime",
    "final_layer",
    "infer_one",
    "initial_layer",
    "layer1",
    "layer2",
    "layer3",
]
