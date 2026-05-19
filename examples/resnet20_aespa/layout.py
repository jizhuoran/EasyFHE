import math
from dataclasses import dataclass

import easyfhe.fhe as fhe

try:
    from .fhe_state import reduce_noise_to_one, rescale_one_level
except ImportError:
    from fhe_state import reduce_noise_to_one, rescale_one_level

__all__ = [
    "broadcast_slot_sum",
    "downsample1024to256",
    "downsample256to64",
    "sum_adjacent_slots",
    "sum_channel_groups",
]


@dataclass(frozen=True)
class _DownsampleSpec:
    spatial_size: int
    out_spatial_size: int
    row_mask_prefix: str
    row_width: int
    row_count: int
    row_rotate: int
    include_gen8: bool
    initial_rescale: str
    rescale_before_fold: bool
    rescale_after_fold: bool


def _merge_fullpack(c1, c2, cryptoContext, weights):
    if c2 is None:
        return c1
    old_slots = c1.slots
    c1 = fhe.slot_resize(c1, c1.slots * 2, cryptoContext)
    c2 = fhe.slot_resize(c2, c2.slots * 2, cryptoContext)
    second_mask_key = f"mask_second_n_{old_slots}_{c2.slots}"
    return fhe.homo_add(
        fhe.homo_mul_pt(
            c1,
            weights.plaintext(
                f"mask_first_n_{old_slots}_{c1.slots}",
                cryptoContext.L - c1.cur_limbs,
                c1.slots,
                cryptoContext,
            ),
            cryptoContext,
        ),
        fhe.homo_mul_pt(
            c2,
            weights.plaintext(
                second_mask_key,
                cryptoContext.L - c2.cur_limbs,
                c2.slots,
                cryptoContext,
            ),
            cryptoContext,
        ),
        cryptoContext,
    )


def _masked_reduce(cipher, mask_n, rotated, cryptoContext, weights):
    cipher = fhe.homo_mul_pt(
        fhe.homo_add(cipher, rotated, cryptoContext),
        weights.plaintext(
            f"gen_mask_{mask_n}_{cipher.slots}",
            cryptoContext.L - cipher.cur_limbs,
            cipher.slots,
            cryptoContext,
        ),
        cryptoContext,
    )
    return rescale_one_level(cipher, cryptoContext)


def _spatial_reduce(fullpack, cryptoContext, weights, include_gen8, initial_rescale):
    if initial_rescale == "always":
        fullpack = rescale_one_level(fullpack, cryptoContext)
    else:
        fullpack = reduce_noise_to_one(fullpack, cryptoContext)
    fullpack = _masked_reduce(fullpack, 2, fhe.homo_rotate(fullpack, 1, cryptoContext), cryptoContext, weights)
    fullpack = _masked_reduce(fullpack, 4, fhe.homo_rotate(fullpack, 2, cryptoContext), cryptoContext, weights)
    if include_gen8:
        fullpack = _masked_reduce(fullpack, 8, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext, weights)
        return fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 8, cryptoContext), cryptoContext)
    return fhe.homo_add(fullpack, fhe.homo_rotate(fullpack, 4, cryptoContext), cryptoContext)


def _pack_rows(fullpack, row_mask_prefix, row_width, spatial_size, row_count, row_rotate, cryptoContext, weights):
    rows = None
    for i in range(row_count):
        masked = fhe.homo_mul_pt(
            fullpack,
            weights.plaintext(
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
    return rescale_one_level(rows, cryptoContext)


def _pack_channels(rows, num_channel, spatial_size, out_spatial_size, cryptoContext, weights):
    channels = None
    for i in range(num_channel * 2):
        masked = fhe.homo_mul_pt(
            rows,
            weights.plaintext(
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
    cipher = fhe.homo_add(
        cipher,
        fhe.homo_rotate(fhe.homo_rotate(cipher, -quarter, cryptoContext), -quarter, cryptoContext),
        cryptoContext,
    )
    return cipher


def _downsample_spatial(c1, c2, num_channel, cryptoContext, weights, spec):
    trace = getattr(cryptoContext, "aespa_state_trace", None)
    if trace is not None:
        trace.append(
            {
                "op": f"downsample{spec.spatial_size}to{spec.out_spatial_size}",
                "cur_limbs": int(c1.cur_limbs),
                "noise_deg": int(c1.noise_deg),
                "slots": int(c1.slots),
                "is_ext": bool(c1.is_ext),
                "second_cipher": c2 is not None,
            }
        )
    fullpack = _merge_fullpack(c1, c2, cryptoContext, weights)
    fullpack = _spatial_reduce(
        fullpack,
        cryptoContext,
        weights,
        include_gen8=spec.include_gen8,
        initial_rescale=spec.initial_rescale,
    )
    rows = _pack_rows(
        fullpack,
        spec.row_mask_prefix,
        spec.row_width,
        spec.spatial_size,
        spec.row_count,
        spec.row_rotate,
        cryptoContext,
        weights,
    )
    channels = _pack_channels(rows, num_channel, spec.spatial_size, spec.out_spatial_size, cryptoContext, weights)
    if spec.rescale_before_fold:
        channels = rescale_one_level(channels, cryptoContext)
    channels = _fold_quarters(channels, cryptoContext)
    if spec.rescale_after_fold:
        channels = rescale_one_level(channels, cryptoContext)
    return fhe.slot_resize(channels, channels.slots // 4, cryptoContext)


def downsample1024to256(c1, c2, num_channel, cryptoContext, weights):
    return _downsample_spatial(
        c1,
        c2,
        num_channel,
        cryptoContext,
        weights,
        _DownsampleSpec(
            spatial_size=1024,
            out_spatial_size=256,
            row_mask_prefix="mask_first_n_mod",
            row_width=16,
            row_count=16,
            row_rotate=48,
            include_gen8=True,
            initial_rescale="if_needed",
            rescale_before_fold=True,
            rescale_after_fold=False,
        ),
    )


def downsample256to64(c1, c2, num_channel, cryptoContext, weights):
    return _downsample_spatial(
        c1,
        c2,
        num_channel,
        cryptoContext,
        weights,
        _DownsampleSpec(
            spatial_size=256,
            out_spatial_size=64,
            row_mask_prefix="mask_first_n_mod2",
            row_width=8,
            row_count=32,
            row_rotate=24,
            include_gen8=False,
            initial_rescale="if_needed",
            rescale_before_fold=False,
            rescale_after_fold=True,
        ),
    )


def sum_adjacent_slots(input, slots, cryptoContext):
    _require_power_of_two(slots, "slots")
    result = input.deep_copy()
    for i in range(int(math.log2(slots))):
        result = fhe.homo_add(result, fhe.homo_rotate(result, 2 ** i, cryptoContext), cryptoContext)
    return result


def sum_channel_groups(input, group_size, num_groups, cryptoContext):
    _require_power_of_two(num_groups, "num_groups")
    result = input.deep_copy()
    for i in range(int(math.log2(num_groups))):
        result = fhe.homo_add(result, fhe.homo_rotate(result, group_size * (2 ** i), cryptoContext), cryptoContext)
    return result


def broadcast_slot_sum(input, slots, cryptoContext):
    return fhe.homo_rotate(sum_adjacent_slots(input, slots, cryptoContext), -slots + 1, cryptoContext)


def _require_power_of_two(value, name):
    if value <= 0 or value & (value - 1) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value}")
