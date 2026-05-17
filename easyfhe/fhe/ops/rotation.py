import easyfhe as torch
import numpy as np

from . import kernels as F
from ..runtime.instrumentation import run_instrumented_op


def _cipher_automorphism(in0, index, cryptoContext):
    norm_index = cryptoContext.norm_rot_index(index)
    limbs = in0.cur_limbs if in0.is_ext == False else in0.cur_limbs + cryptoContext.K
    cv = [F.cv_automorphism_transform(cv, limbs, norm_index, cryptoContext) for cv in in0.cv]
    return in0.cipher_like(cv)


def cipher_automorphism(in0, index, cryptoContext):
    return run_instrumented_op(cryptoContext, "cipher_automorphism", _cipher_automorphism_public, in0, index, cryptoContext)


def _cipher_automorphism_public(in0, index, cryptoContext):
    return _cipher_automorphism(in0, index, cryptoContext)


def homo_rotate(in0, index, cryptoContext):
    return run_instrumented_op(cryptoContext, "homo_rotate", _homo_rotate, in0, index, cryptoContext)


def _homo_rotate(in0, index, cryptoContext):
    if index == 0:
        return in0.deep_copy()
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.get_rotation_key(norm_index)
    special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(index, cryptoContext.L)
    res = in0.cipher_like(F.cv_keyswitch(in0.cv[1], in0.cur_limbs, special_mod_start, swk[0], swk[1], cryptoContext))
    res.cv[0] = F.cv_add(in0.cv[0], res.cv[0], cryptoContext.moduliQ, in0.cur_limbs)
    res = _cipher_automorphism(res, index, cryptoContext)
    return res


def _fast_rotation_key_product(digits, index, cryptoContext):
    if not digits.is_ext:
        raise ValueError("fast rotation key product: expected ext digits")
    norm_index = cryptoContext.norm_rot_index(index)
    swk = cryptoContext.get_rotation_key(norm_index)
    special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(index, cryptoContext.L)
    sum_mult = F.cv_innerproduct(
        digits.cv[0].reshape(-1),
        curr_limbs=digits.cur_limbs,
        special_mod_start=special_mod_start,
        context=cryptoContext,
        swk_bx=swk[0],
        swk_ax=swk[1],
    )
    return digits.cipher_like(sum_mult, is_ext=True)


def _key_switch_P_ext(cipher, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "key_switch_P_ext",
        _key_switch_P_ext_impl,
        cipher,
        cryptoContext,
    )


def _key_switch_P_ext_impl(cipher, cryptoContext):
    if cipher.is_ext:
        raise ValueError("key_switch_P_ext: expected non-ext cipher")
    cv = [
        _scale_to_P_ext(cv, cipher, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=True)


def _modup_to_ext(cipher, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "modup_to_ext",
        _modup_to_ext_impl,
        cipher,
        cryptoContext,
    )


def _modup_to_ext_impl(cipher, cryptoContext):
    if cipher.is_ext:
        raise ValueError("modup_to_ext: expected non-ext cipher")
    if len(cipher.cv) < 2:
        raise ValueError(f"modup_to_ext: expected at least two components, got {len(cipher.cv)}")
    cv = [F.cv_modup(cipher.cv[1], cipher.cur_limbs, cryptoContext)]
    return cipher.cipher_like(cv, is_ext=True)


def _moddown_from_ext(cipher, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "moddown_from_ext",
        _moddown_from_ext_impl,
        cipher,
        cryptoContext,
    )


def moddown_from_ext(cipher, cryptoContext):
    return _moddown_from_ext(cipher, cryptoContext)


def _moddown_from_ext_impl(cipher, cryptoContext):
    if not cipher.is_ext:
        raise ValueError("moddown_from_ext: expected ext cipher")
    cv = [
        F.cv_moddown(cv, cipher.cur_limbs, cryptoContext)
        for cv in cipher.cv
    ]
    return cipher.cipher_like(cv, is_ext=False)


def fast_rotate(cipher, offsets, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "fast_rotate",
        _fast_rotate,
        cipher,
        offsets,
        cryptoContext,
    )


def fast_rotate_batch(cipher, offsets, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "fast_rotate_batch",
        _fast_rotate_batch,
        cipher,
        offsets,
        cryptoContext,
    )


def fast_rotate_ext_batch(cipher, offsets, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "fast_rotate_ext_batch",
        _fast_rotate_ext_batch,
        cipher,
        offsets,
        cryptoContext,
    )


def _fast_rotate_ext_batch(cipher, offsets, cryptoContext):
    if cipher is None or cipher.is_ext:
        raise ValueError("fast_rotate_ext_batch: expected a non-ext cipher")
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("fast_rotate_ext_batch: expected at least one offset")
    if cryptoContext.device == "cuda":
        return _fast_rotate_ext_batch_cuda(cipher, offsets, cryptoContext)
    return _fast_rotate_ext_batch_slow(cipher, offsets, cryptoContext)


def _fast_rotate_ext_batch_slow(cipher, offsets, cryptoContext):
    digits = _modup_to_ext(cipher, cryptoContext)
    return _pack_ciphers([
        _fast_rotate_ext(digits, cipher, index, cryptoContext)
        for index in offsets
    ])


def _fast_rotate_ext_profile_name(cryptoContext, suffix):
    prefix = getattr(cryptoContext, "_fast_rotate_ext_profile_prefix", "")
    return f"{prefix}fast_rotate_ext_batch_{suffix}"


def _fast_rotate_ext_batch_cuda(cipher, offsets, cryptoContext):
    digits = run_instrumented_op(
        cryptoContext,
        _fast_rotate_ext_profile_name(cryptoContext, "modup"),
        _modup_to_ext,
        cipher,
        cryptoContext,
    )
    active_limbs = cipher.cur_limbs + cryptoContext.K
    batch_size = len(offsets)
    key_product_bx = torch.empty(
        (batch_size, active_limbs, cryptoContext.N),
        dtype=torch.uint64,
        device=cryptoContext.device,
    )
    key_product_ax = torch.empty(
        (batch_size, active_limbs, cryptoContext.N),
        dtype=torch.uint64,
        device=cryptoContext.device,
    )

    def key_product_loop():
        for batch_idx, index in enumerate(offsets):
            if index == 0:
                continue
            norm_index = cryptoContext.norm_rot_index(index)
            swk = cryptoContext.get_rotation_key(norm_index)
            special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(index, cryptoContext.L)
            F.cv_innerproduct_write_pair(
                key_product_bx[batch_idx:batch_idx + 1, :, :],
                key_product_ax[batch_idx:batch_idx + 1, :, :],
                digits.cv[0],
                digits.cur_limbs,
                special_mod_start,
                swk[0],
                swk[1],
                cryptoContext,
            )

    run_instrumented_op(
        cryptoContext,
        _fast_rotate_ext_profile_name(cryptoContext, "key_products"),
        key_product_loop,
    )

    def scale_pc():
        return (
            _scale_to_P_ext(cipher.cv[0], cipher, cryptoContext),
            _scale_to_P_ext(cipher.cv[1], cipher, cryptoContext),
        )

    pc0, pc1 = run_instrumented_op(
        cryptoContext,
        _fast_rotate_ext_profile_name(cryptoContext, "scale_pc"),
        scale_pc,
    )
    precomp_maps, offsets_tensor = run_instrumented_op(
        cryptoContext,
        _fast_rotate_ext_profile_name(cryptoContext, "precompute_maps"),
        _batch_precompute_auto,
        offsets,
        cryptoContext,
    )
    cv = run_instrumented_op(
        cryptoContext,
        _fast_rotate_ext_profile_name(cryptoContext, "finalize"),
        F.cv_fast_rotate_ext_batch_finalize_pair,
        key_product_bx,
        key_product_ax,
        pc0,
        pc1,
        precomp_maps,
        offsets_tensor,
        cipher.cur_limbs,
        cryptoContext,
    )
    return cipher.cipher_like(list(cv), is_ext=True, batch_size=batch_size, cipher_id="assign")


def _fast_rotate_batch_cuda(cipher, offsets, cryptoContext):
    if cipher is None or cipher.is_ext:
        raise ValueError("fast_rotate_batch: expected a non-ext cipher")
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("fast_rotate_batch: expected at least one offset")

    digits = _modup_to_ext(cipher, cryptoContext)
    active_limbs = cipher.cur_limbs + cryptoContext.K
    batch_size = len(offsets)
    key_products = torch.empty(
        (2, batch_size, active_limbs, cryptoContext.N),
        dtype=torch.uint64,
        device=cryptoContext.device,
    )

    for batch_idx, index in enumerate(offsets):
        if index == 0:
            continue
        norm_index = cryptoContext.norm_rot_index(index)
        swk = cryptoContext.get_rotation_key(norm_index)
        special_mod_start = cryptoContext.options.rotation_key_limb_limits.get(index, cryptoContext.L)
        F.cv_innerproduct_write(
            key_products[:, batch_idx:batch_idx + 1, :, :],
            digits.cv[0],
            digits.cur_limbs,
            special_mod_start,
            swk[0],
            swk[1],
            cryptoContext,
        )

    moddown_products = torch.empty(
        (2, batch_size, cipher.cur_limbs, cryptoContext.N),
        dtype=torch.uint64,
        device=cryptoContext.device,
    )
    F.cv_moddown_write(moddown_products, key_products, cipher.cur_limbs, cryptoContext)
    precomp_maps, offsets_tensor = _batch_precompute_auto(offsets, cryptoContext)
    cv = F.cv_fast_rotate_batch_finalize(
        moddown_products,
        cipher.cv[0],
        cipher.cv[1],
        precomp_maps,
        offsets_tensor,
        cryptoContext,
    )
    return cipher.cipher_like(list(cv), batch_size=batch_size, cipher_id="assign")


def _fast_rotate_batch(cipher, offsets, cryptoContext):
    if cipher is not None and not cipher.is_ext and cryptoContext.device == "cuda":
        return _fast_rotate_batch_cuda(cipher, offsets, cryptoContext)
    return _pack_ciphers(_fast_rotate(cipher, offsets, cryptoContext))


def _fast_rotate(cipher, offsets, cryptoContext):
    if cipher is None or cipher.is_ext:
        raise ValueError("fast_rotate: expected a non-ext cipher")
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        return []
    digits = _modup_to_ext(cipher, cryptoContext)
    return [
        _fast_rotate_from_digits(digits, cipher, index, cryptoContext)
        for index in offsets
    ]


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
    for index in offsets:
        if index == 0:
            if zero_map is None:
                zero_map = torch.from_numpy(np.zeros(cryptoContext.N, dtype=np.int32)).to(cryptoContext.device)
            maps.append(zero_map)
        else:
            maps.append(cryptoContext.get_precompute_auto(cryptoContext.norm_rot_index(index)))
    precomp_maps = torch.stack(maps, dim=0)
    offsets_tensor = torch.from_numpy(np.array(offsets, dtype=np.int64)).to(cryptoContext.device)
    cache[key] = (precomp_maps, offsets_tensor)
    return cache[key]


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


def _fast_rotate_from_digits(digits, cipher, index, cryptoContext):
    if index == 0:
        return cipher.deep_copy()

    result = _fast_rotation_key_product(digits, index, cryptoContext)
    result = _moddown_from_ext(result, cryptoContext)
    result.cv[0] = F.cv_add(
        result.cv[0],
        cipher.cv[0],
        cryptoContext.moduliQ,
        cipher.cur_limbs,
        inplace=True,
    )
    return _cipher_automorphism(result, index, cryptoContext)


def _fast_rotate_key_contribution_ext(digits, index, cryptoContext):
    if index == 0:
        raise ValueError("fast rotate key contribution: rotation index cannot be 0")
    result = _fast_rotation_key_product(digits, index, cryptoContext)
    return _cipher_automorphism(result, index, cryptoContext)


def _fast_rotate_ext(digits, cipher, index, cryptoContext):
    if cipher is None or cipher.is_ext:
        raise ValueError("fast_rotate_ext: expected a non-ext cipher")
    if index == 0:
        return _key_switch_P_ext(cipher, cryptoContext)

    result = _fast_rotation_key_product(digits, index, cryptoContext)
    cipher_cv0 = F.cv_mul_scalar(
        cipher.cv[0],
        cryptoContext.PModq,
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
        cipher.cur_limbs,
    )
    result.cv[0] = F.cv_add(
        result.cv[0],
        cipher_cv0,
        cryptoContext.moduliQ,
        cipher.cur_limbs,
        inplace=True,
    )
    return _cipher_automorphism(result, index, cryptoContext)


def double_hoist_rotate_sum(inner_exts, offsets, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "double_hoist_rotate_sum",
        _double_hoist_rotate_sum,
        inner_exts,
        offsets,
        cryptoContext,
    )


def _double_hoist_rotate_sum(inner_exts, offsets, cryptoContext):
    from .arithmetic import homo_add
    from .slots import extract_cv

    inner_exts = tuple(inner_exts)
    offsets = tuple(int(offset) for offset in offsets)
    if len(inner_exts) != len(offsets):
        raise ValueError(
            "double_hoist_rotate_sum: inner_exts and offsets must have the same length, "
            f"got {len(inner_exts)} and {len(offsets)}"
        )
    if not inner_exts:
        raise ValueError("double_hoist_rotate_sum: expected at least one inner ext cipher")

    first_acc = None
    outer_ext = None

    for inner_ext, offset in zip(inner_exts, offsets):
        if not inner_ext.is_ext:
            raise ValueError("double_hoist_rotate_sum: expected ext inner ciphers")
        if len(inner_ext.cv) < 2:
            raise ValueError("double_hoist_rotate_sum: expected two-component inner ciphers")

        if offset == 0:
            inner_ext_cv0 = extract_cv(inner_ext, 0, cryptoContext)
            first = _moddown_from_ext(inner_ext_cv0, cryptoContext)
            c1_ext = extract_cv(inner_ext, 1, cryptoContext, append_zeros=True)
        else:
            inner = _moddown_from_ext(inner_ext, cryptoContext)
            inner_cv0 = extract_cv(inner, 0, cryptoContext)
            first = _cipher_automorphism(inner_cv0, offset, cryptoContext)

            inner_digits = _modup_to_ext(inner, cryptoContext)
            c1_ext = _fast_rotate_key_contribution_ext(
                inner_digits,
                offset,
                cryptoContext,
            )

        first_acc = first if first_acc is None else homo_add(first_acc, first, cryptoContext)
        outer_ext = c1_ext if outer_ext is None else homo_add(outer_ext, c1_ext, cryptoContext)

    outer = _moddown_from_ext(outer_ext, cryptoContext)
    first_full_cv = extract_cv(first_acc, 0, cryptoContext, append_zeros=True)
    return homo_add(outer, first_full_cv, cryptoContext)
