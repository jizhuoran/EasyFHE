import easyfhe as torch
import numpy as np

from ..ciphertext import Cipher
from . import kernels as F


def _preserve_component_capacity(template, active):
    if not hasattr(template, "shape") or not hasattr(active, "shape"):
        return active
    if active.dim() < 2 or template.dim() < 2:
        return active
    if active.shape[-1] != template.shape[-1]:
        return active
    capacity = int(template.shape[-2])
    active_limbs = int(active.shape[-2])
    if active_limbs > capacity:
        return active
    desired_shape = tuple(active.shape[:-2]) + (capacity, active.shape[-1])
    if tuple(active.shape) == desired_shape:
        return active
    out = active.new_empty(desired_shape)
    out[..., :active_limbs, :] = active
    return out


def homo_rotate(cipher, offset, cryptoContext, addend=None):
    if offset == 0:
        if addend is not None:
            from .arithmetic import homo_add

            return homo_add(addend, cipher, cryptoContext)
        return cipher.deep_copy()
    if addend is not None:
        if addend.is_ext:
            raise ValueError("homo_rotate(addend=...): expected a non-ext addend")
        if addend.cur_limbs != cipher.cur_limbs:
            raise ValueError(
                "homo_rotate(addend=...): addend cur_limbs must match the rotated cipher"
            )
        if len(addend.cv) != 2:
            raise ValueError("homo_rotate(addend=...): expected two ciphertext components")
    swk_bx, swk_ax, special_mod_start = _rotation_key_and_start(offset, cryptoContext)
    norm_index = _norm_rot_index(offset, cryptoContext)
    cv = F.cv_hrot(
        cipher.cv[0],
        cipher.cv[1],
        cipher.cur_limbs,
        special_mod_start,
        swk_bx,
        swk_ax,
        cryptoContext.get_inverse_precompute_auto(norm_index),
        cryptoContext,
        add_bx=None if addend is None else addend.cv[0],
        add_ax=None if addend is None else addend.cv[1],
    )
    return cipher.cipher_like([
        _preserve_component_capacity(cipher.cv[0], cv[0]),
        _preserve_component_capacity(cipher.cv[1], cv[1]),
    ])


def fast_rotate(cipher, offsets, cryptoContext, *, output_ext=False):
    return _fast_rotate_impl(cipher, _normalize_offsets(offsets), cryptoContext, output_ext=bool(output_ext))


def hoisted_mac_sum(cipher, baby_offsets, plaintexts, giant_offset, giant_count, cryptoContext, *, strategy):
    from .fused import fused_grouped_pairwise_mac

    if strategy == "normal":
        baby_rotations = fast_rotate(cipher, baby_offsets, cryptoContext)
    elif strategy in ("ext_normal", "ext_double_hoist"):
        baby_rotations = fast_rotate(cipher, baby_offsets, cryptoContext, output_ext=True)
    else:
        raise ValueError(f"unknown hoisted_mac_sum strategy: {strategy}")

    partial_sums = fused_grouped_pairwise_mac(
        baby_rotations.cipher_like(baby_rotations.cv, slots=plaintexts.slots),
        plaintexts,
        giant_count,
        cryptoContext,
    )
    return giant_rotate_sum(partial_sums, giant_offset, cryptoContext, strategy=strategy)


def giant_rotate_sum(ciphers, offset, cryptoContext, *, strategy="normal"):
    offset = int(offset)
    if isinstance(ciphers, Cipher):
        if ciphers.batch_size <= 0:
            raise ValueError("giant_rotate_sum: expected at least one cipher")
    else:
        ciphers = tuple(ciphers)
        if not ciphers:
            raise ValueError("giant_rotate_sum: expected at least one cipher")

    if strategy == "ext_double_hoist":
        ciphers = _unpack_cipher_batch(ciphers) if isinstance(ciphers, Cipher) else ciphers
        offsets = tuple(index * offset for index in range(len(ciphers)))
        return _double_hoist_rotate_sum(ciphers, offsets, cryptoContext)
    if strategy == "ext_normal":
        if isinstance(ciphers, Cipher):
            ciphers = moddown_from_ext(ciphers, cryptoContext)
        else:
            ciphers = tuple(moddown_from_ext(cipher, cryptoContext) for cipher in ciphers)
    elif strategy != "normal":
        raise ValueError(f"unknown giant_rotate_sum strategy: {strategy}")
    return _normal_giant_rotate_sum(ciphers, offset, cryptoContext)


def moddown_from_ext(cipher, cryptoContext):
    if not cipher.is_ext:
        raise ValueError("moddown_from_ext: expected ext cipher")
    cv = [F.cv_moddown(cv, cipher.cur_limbs, cryptoContext) for cv in cipher.cv]
    return cipher.cipher_like(cv, is_ext=False)


def _fast_rotate_impl(cipher, offsets, cryptoContext, *, output_ext):
    if cipher.is_ext:
        raise ValueError("fast_rotate: expected a non-ext cipher")

    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("fast_rotate: expected at least one offset")

    batch_size = len(offsets)
    digits = _modup_to_ext(cipher, cryptoContext)
    active_limbs = cipher.cur_limbs + cryptoContext.K
    key_products, product_indices = _fast_rotate_key_products(digits, offsets, active_limbs, cryptoContext)
    precomp_maps = _precompute_auto_maps(offsets, cryptoContext)

    if output_ext:
        return _finalize_fast_rotate_ext(
            cipher,
            key_products,
            product_indices,
            precomp_maps,
            active_limbs,
            batch_size,
            cryptoContext,
        )

    return _finalize_fast_rotate_q(
        cipher,
        key_products,
        product_indices,
        precomp_maps,
        batch_size,
        cryptoContext,
    )


def _finalize_fast_rotate_ext(
    cipher,
    key_products,
    product_indices,
    precomp_maps,
    active_limbs,
    batch_size,
    cryptoContext,
):
    cv = F.cv_fast_rotate_ext_batch_finalize_compact(
        key_products,
        product_indices,
        cipher.cv[0],
        cipher.cv[1],
        precomp_maps,
        cipher.cur_limbs,
        active_limbs,
        cryptoContext,
    )
    return cipher.cipher_like(list(cv), is_ext=True, batch_size=batch_size)


def _finalize_fast_rotate_q(
    cipher,
    key_products,
    product_indices,
    precomp_maps,
    batch_size,
    cryptoContext,
):
    moddown_products = torch.empty(
        (2, key_products.shape[1], cipher.cur_limbs, cryptoContext.N),
        dtype=torch.uint64,
        device=cryptoContext.device,
    )
    if key_products.shape[1] != 0:
        F.cv_moddown_write(moddown_products, key_products, cipher.cur_limbs, cryptoContext)
    cv = F.cv_fast_rotate_batch_finalize_compact(
        moddown_products,
        product_indices,
        cipher.cv[0],
        cipher.cv[1],
        precomp_maps,
        cipher.cur_limbs,
        cryptoContext,
    )
    return cipher.cipher_like(
        [
            _preserve_component_capacity(cipher.cv[0], cv[0]),
            _preserve_component_capacity(cipher.cv[1], cv[1]),
        ],
        batch_size=batch_size,
    )


def _fast_rotate_key_products(digits, offsets, active_limbs, cryptoContext):
    nonzero_offsets, product_indices = _batch_rotation_product_plan(offsets, cryptoContext)
    if not nonzero_offsets:
        return torch.empty(
            (2, 0, active_limbs, cryptoContext.N),
            dtype=torch.uint64,
            device=cryptoContext.device,
        ), product_indices

    swk_bxs, swk_axs, starts = _batch_rotation_keys_and_starts(tuple(nonzero_offsets), cryptoContext)
    key_products = F.cv_innerproduct_broadcast_cipher_pair(
        digits.cv[0],
        digits.cur_limbs,
        starts,
        swk_bxs,
        swk_axs,
        cryptoContext,
    )
    return key_products, product_indices


def _batch_rotation_product_plan(offsets, cryptoContext):
    cache = getattr(cryptoContext, "_fast_rotate_product_plan_cache", None)
    if cache is None:
        cache = {}
        cryptoContext._fast_rotate_product_plan_cache = cache
    key = (tuple(offsets), cryptoContext.device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    product_index_values = []
    nonzero_offsets = []
    for offset in offsets:
        if offset == 0:
            product_index_values.append(-1)
        else:
            product_index_values.append(len(nonzero_offsets))
            nonzero_offsets.append(offset)

    product_indices = torch.from_numpy(np.array(product_index_values, dtype=np.int64)).to(cryptoContext.device)
    cache[key] = (tuple(nonzero_offsets), product_indices)
    return cache[key]


def _batch_rotation_keys_and_starts(offsets, cryptoContext):
    cache = getattr(cryptoContext, "_fast_rotate_key_product_cache", None)
    if cache is None:
        cache = {}
        cryptoContext._fast_rotate_key_product_cache = cache
    key = (tuple(offsets), cryptoContext.device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    swk_bxs = []
    swk_axs = []
    special_mod_starts = []
    for offset in offsets:
        swk_bx, swk_ax, special_mod_start = _rotation_key_and_start(offset, cryptoContext)
        swk_bxs.append(swk_bx)
        swk_axs.append(swk_ax)
        special_mod_starts.append(special_mod_start)

    starts = torch.from_numpy(np.array(special_mod_starts, dtype=np.int64)).to(cryptoContext.device)
    cache[key] = (swk_bxs, swk_axs, starts)
    return cache[key]


def _rotation_key_and_start(offset, cryptoContext):
    norm_index = _norm_rot_index(offset, cryptoContext)
    swk_bx, swk_ax = cryptoContext.get_rotation_key(norm_index)
    special_mod_start = cryptoContext.rotation_key_limb_limits.get(norm_index, cryptoContext.L)
    return swk_bx, swk_ax, special_mod_start


def _norm_rot_index(offset, cryptoContext):
    offset = int(offset)
    if offset < 0:
        return int((int(cryptoContext.N) // 2) + offset)
    return offset



def _normalize_offsets(offsets):
    if isinstance(offsets, (int, np.integer)):
        return (int(offsets),)
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("fast_rotate: expected at least one offset")
    return offsets


def _normal_giant_rotate_sum(ciphers, offset, cryptoContext):
    from .arithmetic import homo_add

    ciphers = _unpack_cipher_batch(ciphers) if isinstance(ciphers, Cipher) else tuple(ciphers)
    if len(ciphers) == 1:
        return ciphers[0]

    result = ciphers[-1]
    for index in range(len(ciphers) - 2, -1, -1):
        if offset != 0:
            result = homo_rotate(result, offset, cryptoContext, addend=ciphers[index])
        else:
            result = homo_add(ciphers[index], result, cryptoContext)
    return result


def _unpack_cipher_batch(cipher):
    if cipher.batch_size == 1 and cipher.cv[0].dim() < 3:
        return (cipher,)
    if cipher.cv[0].dim() < 3:
        raise ValueError("expected batched cipher components")
    return tuple(
        cipher.cipher_like(
            [component[index] for component in cipher.cv],
            batch_size=1,
        )
        for index in range(int(cipher.batch_size))
    )


def _double_hoist_rotate_sum(inner_exts, offsets, cryptoContext):
    from .arithmetic import homo_add
    from .slots import extract_cv

    first_acc = None
    outer_ext = None
    for inner_ext, offset in zip(inner_exts, offsets):
        if offset == 0:
            inner_ext_cv0 = extract_cv(inner_ext, 0, cryptoContext)
            first = moddown_from_ext(inner_ext_cv0, cryptoContext)
            c1_ext = extract_cv(inner_ext, 1, cryptoContext, append_zeros=True)
        else:
            inner = moddown_from_ext(inner_ext, cryptoContext)
            inner_cv0 = extract_cv(inner, 0, cryptoContext)
            first = _cipher_automorphism(inner_cv0, offset, cryptoContext)
            inner_digits = _modup_to_ext(inner, cryptoContext)
            c1_ext = _fast_rotate_key_contribution_ext(inner_digits, offset, cryptoContext)

        first_acc = first if first_acc is None else homo_add(first_acc, first, cryptoContext)
        outer_ext = c1_ext if outer_ext is None else homo_add(outer_ext, c1_ext, cryptoContext)

    outer = moddown_from_ext(outer_ext, cryptoContext)
    first_full_cv = extract_cv(first_acc, 0, cryptoContext, append_zeros=True)
    return homo_add(outer, first_full_cv, cryptoContext)


def _modup_to_ext(cipher, cryptoContext):
    if cipher.is_ext:
        raise ValueError("modup_to_ext: expected non-ext cipher")
    if len(cipher.cv) < 2:
        raise ValueError(f"modup_to_ext: expected at least two components, got {len(cipher.cv)}")
    return cipher.cipher_like([F.cv_modup(cipher.cv[1], cipher.cur_limbs, cryptoContext)], is_ext=True)


def _fast_rotate_key_contribution_ext(digits, offset, cryptoContext):
    swk_bx, swk_ax, special_mod_start = _rotation_key_and_start(offset, cryptoContext)
    result = digits.cipher_like(
        F.cv_innerproduct(
            digits.cv[0].reshape(-1),
            curr_limbs=digits.cur_limbs,
            special_mod_start=special_mod_start,
            context=cryptoContext,
            swk_bx=swk_bx,
            swk_ax=swk_ax,
        ),
        is_ext=True,
    )
    return _cipher_automorphism(result, offset, cryptoContext)


def _cipher_automorphism(cipher, offset, cryptoContext):
    norm_index = _norm_rot_index(offset, cryptoContext)
    limbs = cipher.cur_limbs + (cryptoContext.K if cipher.is_ext else 0)
    cv = [F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext) for cv in cipher.cv]
    return cipher.cipher_like(cv)


def _precompute_auto_maps(offsets, cryptoContext):
    cache = getattr(cryptoContext, "_precompute_auto_maps_cache", None)
    if cache is None:
        cache = {}
        cryptoContext._precompute_auto_maps_cache = cache
    key = (tuple(offsets), cryptoContext.device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    maps = []
    for offset in offsets:
        if offset == 0:
            maps.append(_zero_precompute_auto_map(cryptoContext))
        else:
            maps.append(cryptoContext.get_precompute_auto(_norm_rot_index(offset, cryptoContext)))
    precomp_maps = torch.stack(maps, dim=0)
    cache[key] = precomp_maps
    return cache[key]


def _zero_precompute_auto_map(cryptoContext):
    cached = getattr(cryptoContext, "_zero_precompute_auto_map", None)
    if cached is None:
        cached = torch.from_numpy(np.zeros(cryptoContext.N, dtype=np.int32)).to(cryptoContext.device)
        cryptoContext._zero_precompute_auto_map = cached
    return cached


def _batch_item(cipher, index):
    if cipher.batch_size == 1 and cipher.cv[0].dim() == 2:
        return cipher
    return cipher.cipher_like([cv[int(index)] for cv in cipher.cv], batch_size=1)


def _pack_ciphers(ciphers):
    ciphers = tuple(ciphers)
    if not ciphers:
        raise ValueError("cannot pack an empty cipher batch")
    first = ciphers[0]
    for idx, cipher in enumerate(ciphers):
        if len(cipher.cv) != len(first.cv):
            raise ValueError(f"cipher batch component count mismatch at index {idx}")
        for field in ("cur_limbs", "scaling_factor", "noise_deg", "slots", "is_ext"):
            if getattr(cipher, field) != getattr(first, field):
                raise ValueError(
                    f"cipher batch {field} mismatch at index {idx}: "
                    f"{getattr(cipher, field)} != {getattr(first, field)}"
                )
    cv = [
        torch.stack([cipher.cv[component] for cipher in ciphers], dim=0)
        for component in range(len(first.cv))
    ]
    return first.cipher_like(cv, batch_size=len(ciphers))
