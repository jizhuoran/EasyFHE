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
        conv3x3_sx,
        initial_conv3x3,
        pointwise_conv,
        pointwise_conv_sx,
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
        conv3x3_sx,
        initial_conv3x3,
        pointwise_conv,
        pointwise_conv_sx,
    )
    from weight_pack import WeightPack


@dataclass
class AespaRuntime:
    ctx: object
    client: object
    weights: WeightPack
    config: object
    bootstrap_material: dict[int, tuple[object, object]]


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
    bootstrap_after_nonlinear: bool = False
    use_sx_layout: bool = False
    copy_per_cipher: int = 2


def _convbn_weight_prefix(block_id, conv_id):
    return f"layer{block_id}-conv{conv_id}bn{conv_id}"


def _conv3x3_kernel_group_key(base, channel_offset, channels):
    if channels <= 0:
        raise ValueError("conv3x3 kernel group requires at least one channel")
    return f"{base}-ch{channel_offset}-{channel_offset + channels - 1}-k"


def _conv3x3_kernel_key(block_id, conv_id, channels, channel_offset=0):
    prefix = _convbn_weight_prefix(block_id, conv_id)
    return _conv3x3_kernel_group_key(prefix, channel_offset, channels)


def _pointwise_kernel_group_key(block_id, conv_id, channels, channel_offset=0):
    if channels <= 0:
        raise ValueError("pointwise kernel group requires at least one channel")
    return f"layer{block_id}dx-conv{conv_id}bn{conv_id}-ch{channel_offset}-{channel_offset + channels - 1}-k1"


def _pointwise_bias_key(block_id, conv_id, bias_offset):
    return f"layer{block_id}dx-conv{conv_id}bn{conv_id}-bias{bias_offset}"


def _pointwise_sx_kernel_key(block_id, conv_id):
    return f"layer{block_id}dx-conv{conv_id}bn{conv_id}-sx"


def _pointwise_sx_bias_key(block_id, conv_id):
    return f"layer{block_id}dx-conv{conv_id}bn{conv_id}-bias-sx"


def _bootstrap_material(rt):
    log_bs_slots = int(rt.config.log_bs_slots[0])
    return rt.bootstrap_material[log_bs_slots]


def _bootstrap(rt, input, L0):
    constants, plan = _bootstrap_material(rt)
    return bs.bootstrap(
        input,
        rt.ctx,
        constants,
        plan,
        L0=L0,
    )


def _bootstrap_and_resize_for_downsample(input, rt):
    boot = _bootstrap(rt, input, rt.ctx.L)
    return fhe.slot_resize(boot, boot.slots << 1, rt.ctx)


def _conv_then_aespa_nonlinear(input, block_id, conv_id, img_width, channels, rot_offset, rt, scale=1):
    res = conv3x3(
        input,
        _conv3x3_kernel_key(block_id, conv_id, channels),
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
        _conv3x3_kernel_key(block_id, 1, in_channels),
        in_img_width,
        1,
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    second_half = conv3x3(
        input,
        _conv3x3_kernel_key(block_id, 1, in_channels, channel_offset=in_channels),
        in_img_width,
        1,
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    return rescale_one_level(first_half, rt.ctx), rescale_one_level(second_half, rt.ctx)


def _downsample_conv_sx(input, block_id, in_img_width, out_channels, first_rot, copy_per_cipher, rt, scale):
    result = conv3x3_sx(
        input,
        f"{_convbn_weight_prefix(block_id, 1)}-sx",
        in_img_width,
        1,
        copy_per_cipher,
        out_channels,
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    return rescale_one_level(result, rt.ctx)


def _projection_input_for_downsample(input, rt):
    if not (rt.ctx.scale_mode == "fixed" and rt.ctx.rescale_policy == "manual"):
        return input
    return fhe.align_to(input, fhe.CipherState(input.state.cur_limbs - 2, input.state.noise_deg), rt.ctx)


def _downsample_projection_pair(input, block_id, in_channels, first_rot, rt, scale):
    input = _projection_input_for_downsample(input, rt)
    first_half = pointwise_conv(
        input,
        _pointwise_kernel_group_key(block_id, 1, in_channels),
        _pointwise_bias_key(block_id, 1, "1"),
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    second_half = pointwise_conv(
        input,
        _pointwise_kernel_group_key(block_id, 1, in_channels, channel_offset=in_channels),
        _pointwise_bias_key(block_id, 1, "2"),
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )
    return first_half, second_half


def _downsample_projection_sx(input, block_id, out_channels, first_rot, copy_per_cipher, rt, scale):
    input = _projection_input_for_downsample(input, rt)
    return pointwise_conv_sx(
        input,
        _pointwise_sx_kernel_key(block_id, 1),
        _pointwise_sx_bias_key(block_id, 1),
        copy_per_cipher,
        out_channels,
        first_rot,
        scale,
        rt.ctx,
        rt.weights,
    )


def _same_shape_residual_block(
    input,
    spec,
    rt,
):
    scale = "scale.one"
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
        _conv3x3_kernel_key(spec.block_id, 2, spec.channels),
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
        res = _bootstrap(rt, res, spec.bootstrap_l0)
    return aespa_nonlinear(res, _convbn_weight_prefix(spec.block_id, 2), rt.ctx, rt.weights, scale)


def _downsample_residual_block(
    input,
    spec,
    rt,
):
    scale_sx = "scale.one"
    scale_dx = "scale.one"

    if spec.use_sx_layout:
        sx0 = _downsample_conv_sx(
            input,
            spec.block_id,
            spec.in_img_width,
            spec.out_channels,
            spec.first_rot,
            spec.copy_per_cipher,
            rt,
            scale_sx,
        )
        sx1 = None
        dx0 = _downsample_projection_sx(
            input,
            spec.block_id,
            spec.out_channels,
            spec.first_rot,
            spec.copy_per_cipher,
            rt,
            scale_dx,
        )
        dx1 = None
    else:
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
        _conv3x3_kernel_key(spec.block_id, 2, spec.out_channels),
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

    if spec.bootstrap_after_nonlinear:
        res = aespa_nonlinear(res, _convbn_weight_prefix(spec.block_id, 2), rt.ctx, rt.weights, scale_dx)
        return _bootstrap(rt, res, spec.bootstrap_l0)

    res = _bootstrap(rt, res, spec.bootstrap_l0)
    return aespa_nonlinear(res, _convbn_weight_prefix(spec.block_id, 2), rt.ctx, rt.weights, scale_dx)


def initial_layer(input, rt):
    res = initial_conv3x3(
        input,
        _conv3x3_kernel_group_key("conv1bn1", 0, 16),
        32,
        1,
        1024,
        "scale.one",
        rt.ctx,
        rt.weights,
    )
    res = rescale_one_level(res, rt.ctx)
    return aespa_nonlinear(res, "conv1bn1", rt.ctx, rt.weights)


def layer1(input, rt):
    res = _same_shape_residual_block(input, _SameShapeBlockSpec(1, 32, 16, -1024), rt)
    res = _same_shape_residual_block(res, _SameShapeBlockSpec(2, 32, 16, -1024), rt)
    return _same_shape_residual_block(res, _SameShapeBlockSpec(3, 32, 16, -1024), rt)


def layer2(input, rt):
    input = _bootstrap_and_resize_for_downsample(input, rt)
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
            bootstrap_l0=rt.ctx.L,
            use_sx_layout=True,
        ),
        rt,
    )
    res = _same_shape_residual_block(res, _SameShapeBlockSpec(5, 16, 32, -256), rt)
    return _same_shape_residual_block(res, _SameShapeBlockSpec(6, 16, 32, -256), rt)


def layer3(input, rt):
    input = _bootstrap_and_resize_for_downsample(input, rt)
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
            bootstrap_after_nonlinear=True,
            use_sx_layout=True,
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
            rt.ctx.L - res.state.cur_limbs,
            res.slots,
            rt.ctx,
        ),
        rt.ctx,
    )
    res = broadcast_slot_sum(res, fc_repeat, rt.ctx)
    res = rescale_one_level(res, rt.ctx)
    weight = rt.weights.plaintext(
        f"fc_{res.slots}",
        rt.ctx.L - res.state.cur_limbs,
        res.slots,
        rt.ctx,
    )
    res = fhe.homo_mul_pt(res, weight, rt.ctx)
    res = rescale_one_level(res, rt.ctx)
    res = sum_channel_groups(res, spatial_size, channels, rt.ctx)

    bias = rt.weights.plaintext(
        f"bias_{res.slots}",
        rt.ctx.L - res.state.cur_limbs,
        res.slots,
        rt.ctx,
    )
    return fhe.homo_add_pt(res, bias, rt.ctx)


def encrypt_input(image_vector, rt):
    return rt.client.encrypt(image_vector, device=rt.ctx.device, scale_deg=1, level=14, slots=16 * 32 * 32)


def infer_encrypted(input_cipher, rt):
    first_layer = initial_layer(input_cipher, rt)
    res_layer1 = layer1(first_layer, rt)
    res_layer2 = layer2(res_layer1, rt)
    res_layer3 = layer3(res_layer2, rt)
    return final_layer(res_layer3, rt)


def infer_one(image_vector, rt):
    return infer_encrypted(encrypt_input(image_vector, rt), rt)


__all__ = [
    "AespaRuntime",
    "encrypt_input",
    "final_layer",
    "infer_encrypted",
    "infer_one",
    "initial_layer",
    "layer1",
    "layer2",
    "layer3",
]
