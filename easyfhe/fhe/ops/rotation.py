import easyfhe as torch
import numpy as np

from . import kernels as F
from . import validation
from .layout import cipher_batch_item, pack_cipher_batch, unpack_cipher_batch
from .metadata import active_limbs


_HOIST_STRATEGIES = {"normal", "ext_normal", "ext_double_hoist"}


# Single-cipher rotation primitives.


def homo_rotate(cipher, offset, context):
    return homo_rotate_add(cipher, offset, context)


def homo_rotate_add(cipher, offset, context, addend=None):
    validation.validate_cipher_op("homo_rotate_add", cipher, require_ext=False, require_components=2)
    if addend is not None:
        validation.validate_binary_cipher_op(
            "homo_rotate_add",
            cipher,
            addend,
            require_ext=False,
            require_components=2,
            require_same_metadata=("slots", "cur_limbs", "scale_degree", "scaling_factor"),
        )

    if offset == 0:
        if addend is not None:
            from .arithmetic import homo_add

            return homo_add(addend, cipher, context)
        return cipher.deep_copy()

    if int(cipher.batch_size) > 1:
        rotated = _batch_same_rotate(cipher, int(offset), context)
        if addend is None:
            return rotated
        from .arithmetic import homo_add

        return homo_add(addend, rotated, context)

    swk_bx, swk_ax, special_mod_start = _rotation_key_and_start(offset, context, cipher.state.cur_limbs)
    norm_index = _norm_rot_index(offset, context)
    cv = F.cv_hrot(
        cipher.cv[0],
        cipher.cv[1],
        cipher.state.cur_limbs,
        special_mod_start,
        swk_bx,
        swk_ax,
        context.get_inverse_precompute_auto(norm_index),
        context,
        add_bx=None if addend is None else addend.cv[0],
        add_ax=None if addend is None else addend.cv[1],
    )
    return cipher.cipher_like(list(cv))


def _batch_same_rotate(cipher, offset, context):
    """Apply one automorphism to every item in a ciphertext batch.

    ``fast_rotate`` broadcasts one input ciphertext over many offsets.  This
    is the complementary schedule used by operator-level batching: many
    ciphertexts all use the same offset and evaluation key.  Repeating Python
    key handles does not duplicate their device storage.
    """

    batch_size = int(cipher.batch_size)
    offsets = (int(offset),) * batch_size
    digits = _modup_to_ext(cipher, context)
    active_limb_count = active_limbs(digits, context)
    nonzero_offsets, key_product_indices = _batch_rotation_product_plan(
        offsets, context
    )
    swk_bxs, swk_axs, starts = _batch_rotation_keys_and_starts(
        nonzero_offsets, context, cipher.state.cur_limbs
    )
    key_product_bx, key_product_ax = F.cv_innerproduct_pairwise(
        digits.cv[0],
        digits.state.cur_limbs,
        starts,
        swk_bxs,
        swk_axs,
        context,
    )
    precomp_maps = _precompute_auto_maps(nonzero_offsets, context)
    cv = F.cv_finalize_fast_rotation_ext(
        key_product_bx,
        key_product_ax,
        key_product_indices,
        cipher.cv[0],
        cipher.cv[1],
        precomp_maps,
        cipher.state.cur_limbs,
        active_limb_count,
        context,
    )
    result_ext = cipher.cipher_like(
        list(cv), is_ext=True, batch_size=batch_size
    )
    return moddown_from_ext(result_ext, context)


# Fast batched rotations.


def fast_rotate(cipher, offsets, context, *, output_ext=False):
    validation.validate_cipher_op("fast_rotate", cipher, require_ext=False)

    if isinstance(offsets, (int, np.integer)):
        offsets = (int(offsets),)
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("fast_rotate: expected at least one offset")
    if all(offset == 0 for offset in offsets):
        raise ValueError("fast_rotate: expected at least one nonzero offset")

    batch_size = len(offsets)
    digits = _modup_to_ext(cipher, context)
    active_limb_count = active_limbs(digits, context)
    nonzero_offsets, key_product_indices = _batch_rotation_product_plan(offsets, context)

    swk_bxs, swk_axs, starts = _batch_rotation_keys_and_starts(
        tuple(nonzero_offsets),
        context,
        cipher.state.cur_limbs,
    )
    key_product_bx, key_product_ax = F.cv_innerproduct_broadcast(
        digits.cv[0],
        digits.state.cur_limbs,
        starts,
        swk_bxs,
        swk_axs,
        context,
    )
    precomp_maps = _precompute_auto_maps(nonzero_offsets, context)

    if output_ext:
        cv = F.cv_finalize_fast_rotation_ext(
            key_product_bx,
            key_product_ax,
            key_product_indices,
            cipher.cv[0],
            cipher.cv[1],
            precomp_maps,
            cipher.state.cur_limbs,
            active_limb_count,
            context,
        )
        return cipher.cipher_like(list(cv), is_ext=True, batch_size=batch_size)

    key_product = cipher.cipher_like(
        [key_product_bx, key_product_ax],
        is_ext=True,
        batch_size=len(nonzero_offsets),
    )
    moddown_product = moddown_from_ext(key_product, context)
    cv = F.cv_finalize_fast_rotation_q(
        moddown_product.cv[0],
        moddown_product.cv[1],
        key_product_indices,
        cipher.cv[0],
        cipher.cv[1],
        precomp_maps,
        cipher.state.cur_limbs,
        context,
    )
    return cipher.cipher_like(list(cv), batch_size=batch_size)

# Hoisted MAC and giant-rotation pipelines.


def _baby_block_specs(baby_offsets, baby_anchor_step):
    baby_offsets = tuple(int(offset) for offset in baby_offsets)
    if not baby_offsets:
        raise ValueError("hoisted_mac_sum: expected at least one baby offset")
    baby_anchor_step = int(baby_anchor_step)
    if baby_anchor_step < 0:
        return baby_offsets, ((0, baby_offsets),)
    if baby_anchor_step < 2:
        raise ValueError(
            "hoisted_mac_sum: baby_anchor_step must be at least two or -1"
        )
    if baby_offsets[0] != 0:
        raise ValueError(
            "hoisted_mac_sum: bounded baby steps require offsets to start at zero"
        )
    local_step = baby_offsets[1] if len(baby_offsets) > 1 else 1
    if local_step <= 0 or any(
        baby_offsets[index] - baby_offsets[index - 1] != local_step
        for index in range(1, len(baby_offsets))
    ):
        raise ValueError(
            "hoisted_mac_sum: anchored baby_offsets must be a positive "
            "arithmetic sequence"
        )
    block_width = baby_anchor_step
    specs = []
    for start in range(0, len(baby_offsets), block_width):
        width = min(block_width, len(baby_offsets) - start)
        specs.append(
            (start, tuple(local * local_step for local in range(width)))
        )
    return baby_offsets, tuple(specs)


def _single_zero_rotation_ext(cipher, anchor_offset, context):
    raw = fast_rotate(
        cipher, (0, int(anchor_offset)), context, output_ext=True
    )
    return cipher_batch_item(raw, 0).deep_copy()


def _create_hoisted_baby_rotations(
    cipher,
    baby_offsets,
    context,
    *,
    strategy,
    baby_anchor_step,
):
    baby_offsets, specs = _baby_block_specs(
        baby_offsets, baby_anchor_step
    )
    output_ext = strategy != "normal"
    blocks = []
    anchor = cipher
    anchor_offset = (
        int(baby_offsets[int(baby_anchor_step)])
        if len(baby_offsets) > int(baby_anchor_step)
        else 0
    )
    for block_index, (start, local_offsets) in enumerate(specs):
        if len(local_offsets) == 1 and local_offsets[0] == 0:
            if output_ext:
                block = _single_zero_rotation_ext(
                    anchor, anchor_offset, context
                )
            else:
                block = pack_cipher_batch((anchor.deep_copy(),))
        else:
            block = fast_rotate(
                anchor, local_offsets, context, output_ext=output_ext
            )
        blocks.append(block)
        if block_index + 1 < len(specs):
            anchor = homo_rotate(anchor, anchor_offset, context)
    return tuple(blocks)


def prepare_hoisted_baby_rotations(
    cipher,
    baby_offsets,
    context,
    *,
    strategy,
    baby_anchor_step=-1,
):
    """Prepare the reusable baby-step basis for hoisted BSGS MACs.

    The returned tuple is an opaque, caller-owned basis accepted by
    :func:`hoisted_mac_sum` through its ``baby_rotations`` argument.  It may
    be borrowed by any number of MACs as long as the source ciphertext,
    level, offsets, anchor schedule, and hoisting strategy stay unchanged.

    Keeping preparation separate from the first MAC lets an application
    enqueue the decomposition before it prepares online plaintext weights,
    while preserving the existing ``return_baby_rotations`` convenience
    path for one-shot callers.
    """

    strategy = _require_hoist_strategy(
        "prepare_hoisted_baby_rotations", strategy
    )
    baby_offsets, _ = _baby_block_specs(
        baby_offsets, baby_anchor_step
    )
    return _create_hoisted_baby_rotations(
        cipher,
        baby_offsets,
        context,
        strategy=strategy,
        baby_anchor_step=int(baby_anchor_step),
    )


def _validate_cached_baby_rotations(
    cipher,
    baby_offsets,
    baby_rotations,
    *,
    strategy,
    baby_anchor_step,
):
    if not isinstance(baby_rotations, (tuple, list)):
        raise TypeError(
            "hoisted_mac_sum: baby_rotations must be a tuple of cipher batches"
        )
    baby_offsets, specs = _baby_block_specs(
        baby_offsets, baby_anchor_step
    )
    if len(baby_rotations) != len(specs):
        raise ValueError("hoisted_mac_sum: cached baby rotation block count mismatch")
    expected_ext = strategy != "normal"
    for block, (_, local_offsets) in zip(
        baby_rotations, specs, strict=True
    ):
        if int(block.batch_size) != len(local_offsets):
            raise ValueError(
                "hoisted_mac_sum: cached baby rotation block size mismatch"
            )
        if block.state != cipher.state or int(block.slots) != int(cipher.slots):
            raise ValueError(
                "hoisted_mac_sum: cached baby rotations must match the input "
                "cipher metadata"
            )
        if bool(block.is_ext) != expected_ext:
            raise ValueError(
                "hoisted_mac_sum: cached baby rotation basis does not match "
                f"strategy={strategy!r}"
            )
    return baby_offsets, specs


def _plaintext_baby_block(
    plaintexts,
    *,
    giant_count,
    baby_count,
    start,
    width,
):
    components = []
    for component in plaintexts.cv:
        shaped = component.reshape(
            int(giant_count), int(baby_count), *component.shape[1:]
        )
        selected = shaped[:, int(start) : int(start + width)]
        components.append(
            selected.reshape(
                int(giant_count) * int(width), *component.shape[1:]
            )
        )
    return plaintexts.cipher_like(
        components, batch_size=int(giant_count) * int(width)
    )


def _grouped_mac_baby_blocks(
    baby_rotations,
    plaintexts,
    giant_count,
    context,
    *,
    baby_count,
    specs,
):
    from .arithmetic import grouped_pairwise_mac, homo_add

    partial_sums = None
    for (start, _), block in zip(specs, baby_rotations, strict=True):
        block_plaintexts = _plaintext_baby_block(
            plaintexts,
            giant_count=giant_count,
            baby_count=baby_count,
            start=start,
            width=int(block.batch_size),
        )
        block_sums = grouped_pairwise_mac(
            block.cipher_like(block.cv, slots=block_plaintexts.slots),
            block_plaintexts,
            giant_count,
            context,
        )
        partial_sums = (
            block_sums
            if partial_sums is None
            else homo_add(partial_sums, block_sums, context)
        )
    return partial_sums


def hoisted_mac_sum(
    cipher,
    baby_offsets,
    plaintexts,
    giant_offset,
    giant_count,
    context,
    *,
    strategy,
    baby_anchor_step=-1,
    baby_rotations=None,
    return_baby_rotations=False,
):
    """Run a hoisted BSGS MAC, optionally borrowing or returning baby steps.

    The two schedule values are ``Bstep=len(baby_offsets)`` and
    ``anchorstep=baby_anchor_step``.  For Bstep 128 and anchorstep 32, anchors
    0/32/64/96 are reached sequentially and four fast rotations each cover
    local offsets 0..31.  The physical stride and anchor rotation are inferred
    from the arithmetic ``baby_offsets`` sequence.  Set
    ``baby_anchor_step=-1`` to retain the traditional single fast batch.

    ``baby_rotations`` is a tuple of cipher batches returned for the same
    input and step parameters.  It is borrowed and never consumed or mutated.
    With ``return_baby_rotations=True``, return ``(result, baby_rotations)`` so
    the caller can reuse all blocks across plaintext chunks or weight pages.

    The default call remains backward compatible and returns only ``result``.
    """
    strategy = _require_hoist_strategy("hoisted_mac_sum", strategy)
    baby_offsets, specs = _baby_block_specs(
        baby_offsets, baby_anchor_step
    )
    if baby_rotations is None:
        baby_rotations = prepare_hoisted_baby_rotations(
            cipher,
            baby_offsets,
            context,
            strategy=strategy,
            baby_anchor_step=baby_anchor_step,
        )
    else:
        baby_offsets, specs = _validate_cached_baby_rotations(
            cipher,
            baby_offsets,
            baby_rotations,
            strategy=strategy,
            baby_anchor_step=baby_anchor_step,
        )
    partial_sums = _grouped_mac_baby_blocks(
        baby_rotations,
        plaintexts,
        giant_count,
        context,
        baby_count=len(tuple(baby_offsets)),
        specs=specs,
    )
    if int(giant_count) == 1:
        result = cipher_batch_item(partial_sums, 0)
        result = moddown_from_ext(result, context) if result.is_ext else result
    elif int(giant_offset) == 0:
        result = _sum_batch_items_without_rotation(partial_sums, context)
    else:
        result = giant_rotate_sum(
            partial_sums, giant_offset, context, strategy=strategy
        )
    if bool(return_baby_rotations):
        return result, baby_rotations
    return result


def hoisted_mac_sum_rescale(
    cipher,
    baby_offsets,
    plaintexts,
    giant_offset,
    giant_count,
    context,
    *,
    strategy,
    baby_anchor_step=-1,
    baby_rotations=None,
    return_baby_rotations=False,
):
    """Run a hoisted MAC sum and consume the single u64 rescale limb.

    The optional baby-rotation input/output follows :func:`hoisted_mac_sum`.
    """

    from .alignment import rescale

    mac_result = hoisted_mac_sum(
        cipher,
        baby_offsets,
        plaintexts,
        giant_offset,
        giant_count,
        context,
        strategy=strategy,
        baby_anchor_step=baby_anchor_step,
        baby_rotations=baby_rotations,
        return_baby_rotations=bool(return_baby_rotations),
    )
    if bool(return_baby_rotations):
        result, used_baby_rotations = mac_result
        return rescale(result, context), used_baby_rotations
    return rescale(mac_result, context)


def giant_rotate_sum(ciphers, offset, context, *, strategy="normal"):
    strategy = _require_hoist_strategy("giant_rotate_sum", strategy)
    offset = int(offset)
    if offset == 0:
        raise ValueError("giant_rotate_sum: offset must be nonzero")
    validation.require_batched_cipher("giant_rotate_sum", ciphers, "ciphers")
    if ciphers.batch_size <= 1:
        raise ValueError(f"giant_rotate_sum: expected batch_size > 1, got {ciphers.batch_size}")

    if strategy == "ext_double_hoist":
        result_ext = ciphers.cipher_like(
            [component[0] for component in ciphers.cv],
            batch_size=1,
        )
        tail_ext = ciphers.cipher_like(
            [component[1:] for component in ciphers.cv],
            batch_size=int(ciphers.batch_size) - 1,
        )
        inner = moddown_from_ext(tail_ext, context)
        inner_digits = _modup_to_ext(inner, context)
        offsets = tuple(
            _double_hoist_giant_key_offset(index * offset, context)
            for index in range(1, int(tail_ext.batch_size) + 1)
        )
        active_limb_count = active_limbs(inner_digits, context)
        swk_bxs, swk_axs, starts = _batch_rotation_keys_and_starts(offsets, context, inner.state.cur_limbs)
        key_product_bx, key_product_ax = F.cv_innerproduct_pairwise(
            inner_digits.cv[0],
            inner_digits.state.cur_limbs,
            starts,
            swk_bxs,
            swk_axs,
            context,
        )
        precomp_maps = _precompute_auto_maps(offsets, context)
        cv = F.cv_double_hoist_giant_sum_ext(
            result_ext.cv[0][0],
            result_ext.cv[1][0],
            key_product_bx,
            key_product_ax,
            inner.cv[0],
            precomp_maps,
            inner.state.cur_limbs,
            active_limb_count,
            context,
        )
        result_ext = result_ext.cipher_like(list(cv), is_ext=True, batch_size=1)
        return moddown_from_ext(result_ext, context)
    if strategy == "ext_normal":
        ciphers = moddown_from_ext(ciphers, context)

    ciphers = unpack_cipher_batch(ciphers)
    result = ciphers[-1]
    for index in range(len(ciphers) - 2, -1, -1):
        result = homo_rotate_add(result, offset, context, addend=ciphers[index])
    return result


def _require_hoist_strategy(op_name, strategy):
    strategy = str(strategy)
    if strategy not in _HOIST_STRATEGIES:
        raise ValueError(
            f"{op_name}: strategy must be one of {sorted(_HOIST_STRATEGIES)}, got {strategy!r}"
        )
    return strategy


def _sum_batch_items_without_rotation(ciphers, context):
    from .arithmetic import sum_cipher_batch

    validation.require_batched_cipher("_sum_batch_items_without_rotation", ciphers, "ciphers")
    result = sum_cipher_batch(ciphers, context)
    return moddown_from_ext(result, context) if result.is_ext else result


# Ext-domain conversion.


def moddown_from_ext(cipher, context):
    validation.validate_cipher_op("moddown_from_ext", cipher, require_ext=True)
    cv = [F.cv_moddown(cv, cipher.state.cur_limbs, context) for cv in cipher.cv]
    return cipher.cipher_like(cv, is_ext=False)


def _modup_to_ext(cipher, context):
    validation.validate_cipher_op("modup_to_ext", cipher, require_ext=False)
    if len(cipher.cv) < 2:
        raise ValueError(f"modup_to_ext: expected at least two components, got {len(cipher.cv)}")
    return cipher.cipher_like([F.cv_modup(cipher.cv[1], cipher.state.cur_limbs, context)], is_ext=True)


# Rotation-key lookup and index tables.


def _batch_rotation_product_plan(offsets, context):
    product_index_values = []
    nonzero_offsets = []
    for offset in offsets:
        if offset == 0:
            product_index_values.append(-1)
        else:
            product_index_values.append(len(nonzero_offsets))
            nonzero_offsets.append(offset)

    key_product_indices = torch.from_numpy(np.array(product_index_values, dtype=np.int64)).to(context.device)
    return tuple(nonzero_offsets), key_product_indices


def _batch_rotation_keys_and_starts(offsets, context, cur_limbs):
    swk_bxs = []
    swk_axs = []
    special_mod_starts = []
    for offset in offsets:
        swk_bx, swk_ax, special_mod_start = _rotation_key_and_start(offset, context, cur_limbs)
        swk_bxs.append(swk_bx)
        swk_axs.append(swk_ax)
        special_mod_starts.append(special_mod_start)

    starts = torch.from_numpy(np.array(special_mod_starts, dtype=np.int64)).to(context.device)
    return swk_bxs, swk_axs, starts


def _rotation_key_and_start(offset, context, cur_limbs):
    norm_index = _norm_rot_index(offset, context)
    if hasattr(context, "get_rotation_key_for_limbs"):
        (swk_bx, swk_ax), special_mod_start = context.get_rotation_key_for_limbs(norm_index, cur_limbs)
    else:
        swk_bx, swk_ax = context.get_rotation_key(norm_index)
        special_mod_start = context.rotation_key_limb_limits.get(norm_index, context.L)
    return swk_bx, swk_ax, special_mod_start


def _norm_rot_index(offset, context):
    offset = int(offset)
    if offset < 0:
        return int((int(context.N) // 2) + offset)
    return offset


def _double_hoist_giant_key_offset(offset, context):
    offset = int(offset)
    half_ring = int(context.N) // 2
    if offset > half_ring:
        return offset % half_ring
    return offset


def _precompute_auto_maps(offsets, context):
    cache = context.precompute_auto_maps_cache
    key = (tuple(offsets), context.device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    maps = []
    for offset in offsets:
        maps.append(context.get_precompute_auto(_norm_rot_index(offset, context)))
    precomp_maps = torch.stack(maps, dim=0)
    cache[key] = precomp_maps
    return cache[key]
