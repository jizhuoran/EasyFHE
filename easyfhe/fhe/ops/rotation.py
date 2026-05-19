import easyfhe as torch
import numpy as np

from ..ciphertext import Cipher
from . import kernels as F


def homo_rotate(cipher, offset, cryptoContext):
    if offset == 0:
        return cipher.deep_copy()
    norm_index = cryptoContext.norm_rot_index(offset)
    swk = cryptoContext.get_rotation_key(norm_index)
    special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(offset, cryptoContext.L)
    result = cipher.cipher_like(
        F.cv_keyswitch(
            cipher.cv[1],
            cipher.cur_limbs,
            special_mod_start,
            swk[0],
            swk[1],
            cryptoContext,
        )
    )
    result.cv[0] = F.cv_add(cipher.cv[0], result.cv[0], cryptoContext.moduliQ, cipher.cur_limbs)
    return _cipher_automorphism(result, offset, cryptoContext)


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
    ciphers = tuple(ciphers)
    offset = int(offset)
    if not ciphers:
        raise ValueError("giant_rotate_sum: expected at least one cipher")

    if strategy == "ext_double_hoist":
        offsets = tuple(index * offset for index in range(len(ciphers)))
        return _double_hoist_rotate_sum(ciphers, offsets, cryptoContext)
    if strategy == "ext_normal":
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
    key_products = torch.empty(
        (2, batch_size, active_limbs, cryptoContext.N),
        dtype=torch.uint64,
        device=cryptoContext.device,
    )
    for batch_idx, offset in enumerate(offsets):
        if offset == 0:
            continue
        norm_index = cryptoContext.norm_rot_index(offset)
        swk = cryptoContext.get_rotation_key(norm_index)
        special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(offset, cryptoContext.L)
        F.cv_innerproduct_write(
            key_products[:, batch_idx:batch_idx + 1, :, :],
            digits.cv[0],
            digits.cur_limbs,
            special_mod_start,
            swk[0],
            swk[1],
            cryptoContext,
        )

    precomp_maps, offsets_tensor = _batch_precompute_auto(offsets, cryptoContext)
    if output_ext:
        cv = F.cv_fast_rotate_ext_batch_finalize(
            key_products,
            _scale_to_P_ext(cipher.cv[0], cipher, cryptoContext),
            _scale_to_P_ext(cipher.cv[1], cipher, cryptoContext),
            precomp_maps,
            offsets_tensor,
            cipher.cur_limbs,
            cryptoContext,
        )
        return cipher.cipher_like(list(cv), is_ext=True, batch_size=batch_size, cipher_id="assign")

    moddown_products = torch.empty(
        (2, batch_size, cipher.cur_limbs, cryptoContext.N),
        dtype=torch.uint64,
        device=cryptoContext.device,
    )
    F.cv_moddown_write(moddown_products, key_products, cipher.cur_limbs, cryptoContext)
    cv = F.cv_fast_rotate_batch_finalize(
        moddown_products,
        cipher.cv[0],
        cipher.cv[1],
        precomp_maps,
        offsets_tensor,
        cryptoContext,
    )
    return cipher.cipher_like(list(cv), batch_size=batch_size, cipher_id="assign")


def _normalize_offsets(offsets):
    if isinstance(offsets, (int, np.integer)):
        return (int(offsets),)
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("fast_rotate: expected at least one offset")
    return offsets


def _normal_giant_rotate_sum(ciphers, offset, cryptoContext):
    from .arithmetic import homo_add

    if len(ciphers) == 1:
        return ciphers[0]

    result = ciphers[-1]
    for index in range(len(ciphers) - 2, -1, -1):
        if offset != 0:
            result = homo_rotate(result, offset, cryptoContext)
        result = homo_add(ciphers[index], result, cryptoContext)
    return result


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
    norm_index = cryptoContext.norm_rot_index(offset)
    swk = cryptoContext.get_rotation_key(norm_index)
    special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(offset, cryptoContext.L)
    result = digits.cipher_like(
        F.cv_innerproduct(
            digits.cv[0].reshape(-1),
            curr_limbs=digits.cur_limbs,
            special_mod_start=special_mod_start,
            context=cryptoContext,
            swk_bx=swk[0],
            swk_ax=swk[1],
        ),
        is_ext=True,
    )
    return _cipher_automorphism(result, offset, cryptoContext)


def _cipher_automorphism(cipher, offset, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(offset)
    limbs = cipher.cur_limbs + (cryptoContext.K if cipher.is_ext else 0)
    cv = [F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext) for cv in cipher.cv]
    return cipher.cipher_like(cv)


def _scale_to_P_ext(cv, cipher, cryptoContext):
    return torch.cat((
        F.cv_mul_scalar(cv, cryptoContext.PModq, cryptoContext.moduliQ, cryptoContext.q_mu, cipher.cur_limbs),
        torch.zeros(
            (cryptoContext.K << cryptoContext.logN),
            dtype=torch.uint64,
            device=cryptoContext.device,
        ).reshape(-1, cryptoContext.N),
    ), dim=0)


def _batch_precompute_auto(offsets, cryptoContext):
    cache = getattr(cryptoContext, "_precompute_auto_batch_cache", None)
    if cache is None:
        cache = {}
        cryptoContext._precompute_auto_batch_cache = cache
    key = (tuple(offsets), cryptoContext.device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    zero_map = None
    maps = []
    for offset in offsets:
        if offset == 0:
            if zero_map is None:
                zero_map = torch.from_numpy(np.zeros(cryptoContext.N, dtype=np.int32)).to(cryptoContext.device)
            maps.append(zero_map)
        else:
            maps.append(cryptoContext.get_precompute_auto(cryptoContext.norm_rot_index(offset)))
    precomp_maps = torch.stack(maps, dim=0)
    offsets_tensor = torch.from_numpy(np.array(offsets, dtype=np.int64)).to(cryptoContext.device)
    cache[key] = (precomp_maps, offsets_tensor)
    return cache[key]


def _batch_item(cipher, index):
    if cipher.batch_size == 1 and cipher.cv[0].dim() == 2:
        return cipher
    return cipher.cipher_like([cv[int(index)] for cv in cipher.cv], batch_size=1, cipher_id="assign")


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
    return first.cipher_like(cv, batch_size=len(ciphers), cipher_id="assign")
