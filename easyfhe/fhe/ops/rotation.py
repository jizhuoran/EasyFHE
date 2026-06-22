from enum import Enum, auto

import easyfhe as torch
import numpy as np

from . import kernels as F
from . import validation
from .layout import cipher_batch_item, unpack_cipher_batch


class HoistStrategy(Enum):
    NORMAL = auto()
    EXT_NORMAL = auto()
    EXT_DOUBLE_HOIST = auto()


HOIST_NORMAL = HoistStrategy.NORMAL
HOIST_EXT_NORMAL = HoistStrategy.EXT_NORMAL
HOIST_EXT_DOUBLE_HOIST = HoistStrategy.EXT_DOUBLE_HOIST


# Single-cipher rotation primitives.


def homo_rotate(cipher, offset, cryptoContext):
    return homo_rotate_add(cipher, offset, cryptoContext)


def homo_rotate_add(cipher, offset, cryptoContext, addend=None):
    validation.validate_cipher_op("homo_rotate_add", cipher, require_ext=False, require_components=2)
    if addend is not None:
        validation.validate_binary_cipher_op(
            "homo_rotate_add",
            cipher,
            addend,
            require_ext=False,
            require_components=2,
            require_same_metadata=("slots", "cur_limbs", "noise_deg", "scaling_factor"),
        )

    if offset == 0:
        if addend is not None:
            from .arithmetic import homo_add

            return homo_add(addend, cipher, cryptoContext)
        return cipher.deep_copy()

    swk_bx, swk_ax, special_mod_start = _rotation_key_and_start(offset, cryptoContext, cipher.state.cur_limbs)
    norm_index = _norm_rot_index(offset, cryptoContext)
    cv = F.cv_hrot(
        cipher.cv[0],
        cipher.cv[1],
        cipher.state.cur_limbs,
        special_mod_start,
        swk_bx,
        swk_ax,
        cryptoContext.get_inverse_precompute_auto(norm_index),
        cryptoContext,
        add_bx=None if addend is None else addend.cv[0],
        add_ax=None if addend is None else addend.cv[1],
    )
    return cipher.cipher_like(list(cv))


# Fast batched rotations.


def fast_rotate(cipher, offsets, cryptoContext, *, output_ext=False):
    validation.validate_cipher_op("fast_rotate", cipher, require_ext=False)

    if isinstance(offsets, (int, np.integer)):
        offsets = (int(offsets),)
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("fast_rotate: expected at least one offset")
    if all(offset == 0 for offset in offsets):
        raise ValueError("fast_rotate: expected at least one nonzero offset")

    batch_size = len(offsets)
    digits = _modup_to_ext(cipher, cryptoContext)
    active_limbs = cipher.state.cur_limbs + cryptoContext.K
    nonzero_offsets, key_product_indices = _batch_rotation_product_plan(offsets, cryptoContext)

    swk_bxs, swk_axs, starts = _batch_rotation_keys_and_starts(
        tuple(nonzero_offsets),
        cryptoContext,
        cipher.state.cur_limbs,
    )
    key_product_bx, key_product_ax = F.cv_innerproduct_broadcast(
        digits.cv[0],
        digits.state.cur_limbs,
        starts,
        swk_bxs,
        swk_axs,
        cryptoContext,
    )
    precomp_maps = _precompute_auto_maps(nonzero_offsets, cryptoContext)

    if output_ext:
        cv = F.cv_finalize_fast_rotation_ext(
            key_product_bx,
            key_product_ax,
            key_product_indices,
            cipher.cv[0],
            cipher.cv[1],
            precomp_maps,
            cipher.state.cur_limbs,
            active_limbs,
            cryptoContext,
        )
        return cipher.cipher_like(list(cv), is_ext=True, batch_size=batch_size)

    key_product = cipher.cipher_like(
        [key_product_bx, key_product_ax],
        is_ext=True,
        batch_size=len(nonzero_offsets),
    )
    moddown_product = moddown_from_ext(key_product, cryptoContext)
    cv = F.cv_finalize_fast_rotation_q(
        moddown_product.cv[0],
        moddown_product.cv[1],
        key_product_indices,
        cipher.cv[0],
        cipher.cv[1],
        precomp_maps,
        cipher.state.cur_limbs,
        cryptoContext,
    )
    return cipher.cipher_like(list(cv), batch_size=batch_size)

# Hoisted MAC and giant-rotation pipelines.


def hoisted_mac_sum(cipher, baby_offsets, plaintexts, giant_offset, giant_count, cryptoContext, *, strategy):
    from .arithmetic import grouped_pairwise_mac

    if strategy == HOIST_NORMAL:
        baby_rotations = fast_rotate(cipher, baby_offsets, cryptoContext)
    else:
        baby_rotations = fast_rotate(cipher, baby_offsets, cryptoContext, output_ext=True)

    partial_sums = grouped_pairwise_mac(
        baby_rotations.cipher_like(baby_rotations.cv, slots=plaintexts.slots),
        plaintexts,
        giant_count,
        cryptoContext,
    )
    if int(giant_count) == 1:
        result = cipher_batch_item(partial_sums, 0)
        return moddown_from_ext(result, cryptoContext) if result.is_ext else result
    if int(giant_offset) == 0:
        return _sum_batch_items_without_rotation(partial_sums, cryptoContext)
    return giant_rotate_sum(partial_sums, giant_offset, cryptoContext, strategy=strategy)


def giant_rotate_sum(ciphers, offset, cryptoContext, *, strategy=HOIST_NORMAL):
    offset = int(offset)
    if offset == 0:
        raise ValueError("giant_rotate_sum: offset must be nonzero")
    validation.require_batched_cipher("giant_rotate_sum", ciphers, "ciphers")
    if ciphers.batch_size <= 1:
        raise ValueError(f"giant_rotate_sum: expected batch_size > 1, got {ciphers.batch_size}")

    if strategy == HOIST_EXT_DOUBLE_HOIST:
        result_ext = ciphers.cipher_like(
            [component[0] for component in ciphers.cv],
            batch_size=1,
        )
        tail_ext = ciphers.cipher_like(
            [component[1:] for component in ciphers.cv],
            batch_size=int(ciphers.batch_size) - 1,
        )
        inner = moddown_from_ext(tail_ext, cryptoContext)
        inner_digits = _modup_to_ext(inner, cryptoContext)
        offsets = tuple(index * offset for index in range(1, int(tail_ext.batch_size) + 1))
        active_limbs = inner.state.cur_limbs + cryptoContext.K
        swk_bxs, swk_axs, starts = _batch_rotation_keys_and_starts(offsets, cryptoContext, inner.state.cur_limbs)
        key_product_bx, key_product_ax = F.cv_innerproduct_pairwise(
            inner_digits.cv[0],
            inner_digits.state.cur_limbs,
            starts,
            swk_bxs,
            swk_axs,
            cryptoContext,
        )
        precomp_maps = _precompute_auto_maps(offsets, cryptoContext)
        cv = F.cv_double_hoist_giant_sum_ext(
            result_ext.cv[0][0],
            result_ext.cv[1][0],
            key_product_bx,
            key_product_ax,
            inner.cv[0],
            precomp_maps,
            inner.state.cur_limbs,
            active_limbs,
            cryptoContext,
        )
        result_ext = result_ext.cipher_like(list(cv), is_ext=True, batch_size=1)
        return moddown_from_ext(result_ext, cryptoContext)
    if strategy == HOIST_EXT_NORMAL:
        ciphers = moddown_from_ext(ciphers, cryptoContext)

    ciphers = unpack_cipher_batch(ciphers)
    result = ciphers[-1]
    for index in range(len(ciphers) - 2, -1, -1):
        result = homo_rotate_add(result, offset, cryptoContext, addend=ciphers[index])
    return result


def _sum_batch_items_without_rotation(ciphers, cryptoContext):
    from .primitives import _cipher_add

    validation.require_batched_cipher("_sum_batch_items_without_rotation", ciphers, "ciphers")
    items = unpack_cipher_batch(ciphers)
    result = items[0]
    for item in items[1:]:
        result = _cipher_add(result, item, cryptoContext)
    return moddown_from_ext(result, cryptoContext) if result.is_ext else result


# Ext-domain conversion.


def moddown_from_ext(cipher, cryptoContext):
    validation.validate_cipher_op("moddown_from_ext", cipher, require_ext=True)
    cv = [F.cv_moddown(cv, cipher.state.cur_limbs, cryptoContext) for cv in cipher.cv]
    return cipher.cipher_like(cv, is_ext=False)


def _modup_to_ext(cipher, cryptoContext):
    validation.validate_cipher_op("modup_to_ext", cipher, require_ext=False)
    if len(cipher.cv) < 2:
        raise ValueError(f"modup_to_ext: expected at least two components, got {len(cipher.cv)}")
    return cipher.cipher_like([F.cv_modup(cipher.cv[1], cipher.state.cur_limbs, cryptoContext)], is_ext=True)


# Rotation-key lookup and index tables.


def _batch_rotation_product_plan(offsets, cryptoContext):
    product_index_values = []
    nonzero_offsets = []
    for offset in offsets:
        if offset == 0:
            product_index_values.append(-1)
        else:
            product_index_values.append(len(nonzero_offsets))
            nonzero_offsets.append(offset)

    key_product_indices = torch.from_numpy(np.array(product_index_values, dtype=np.int64)).to(cryptoContext.device)
    return tuple(nonzero_offsets), key_product_indices


def _batch_rotation_keys_and_starts(offsets, cryptoContext, cur_limbs):
    swk_bxs = []
    swk_axs = []
    special_mod_starts = []
    for offset in offsets:
        swk_bx, swk_ax, special_mod_start = _rotation_key_and_start(offset, cryptoContext, cur_limbs)
        swk_bxs.append(swk_bx)
        swk_axs.append(swk_ax)
        special_mod_starts.append(special_mod_start)

    starts = torch.from_numpy(np.array(special_mod_starts, dtype=np.int64)).to(cryptoContext.device)
    return swk_bxs, swk_axs, starts


def _rotation_key_and_start(offset, cryptoContext, cur_limbs):
    norm_index = _norm_rot_index(offset, cryptoContext)
    if hasattr(cryptoContext, "get_rotation_key_for_limbs"):
        (swk_bx, swk_ax), special_mod_start = cryptoContext.get_rotation_key_for_limbs(norm_index, cur_limbs)
    else:
        swk_bx, swk_ax = cryptoContext.get_rotation_key(norm_index)
        special_mod_start = cryptoContext.rotation_key_limb_limits.get(norm_index, cryptoContext.L)
    return swk_bx, swk_ax, special_mod_start


def _norm_rot_index(offset, cryptoContext):
    offset = int(offset)
    if offset < 0:
        return int((int(cryptoContext.N) // 2) + offset)
    return offset


def _precompute_auto_maps(offsets, cryptoContext):
    cache = cryptoContext.precompute_auto_maps_cache
    key = (tuple(offsets), cryptoContext.device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    maps = []
    for offset in offsets:
        maps.append(cryptoContext.get_precompute_auto(_norm_rot_index(offset, cryptoContext)))
    precomp_maps = torch.stack(maps, dim=0)
    cache[key] = precomp_maps
    return cache[key]
