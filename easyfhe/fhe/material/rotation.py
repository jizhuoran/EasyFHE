from __future__ import annotations

from functools import lru_cache
from dataclasses import replace

import numpy as np
import easyfhe as torch

from easyfhe.bs.openfhe.internal.rotations import bootstrap_auto_index_map
from .native_sampler import NativeCppSampleProvider


def mod_inverse(value, modulus):
    old_r, r = int(value), int(modulus)
    old_s, s = 1, 0
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    if old_r != 1:
        raise ValueError(f"{value} is not invertible modulo {modulus}")
    return old_s % modulus


def find_auto_index_2n_complex(rot_index, cycl_order):
    rot_index = int(rot_index)
    cycl_order = int(cycl_order)
    if rot_index == 0:
        return 1
    if rot_index == cycl_order - 1:
        return cycl_order - 1

    generator = mod_inverse(5, cycl_order) if rot_index < 0 else 5
    result = generator
    for _ in range(1, abs(rot_index)):
        result = (result * generator) % cycl_order
    return int(result)


@lru_cache(maxsize=None)
def bit_reverse_indices(ring_dim: int) -> np.ndarray:
    ring_dim = int(ring_dim)
    if ring_dim <= 0 or ring_dim & (ring_dim - 1):
        raise ValueError(f"ring_dim must be a positive power of two, got {ring_dim}")

    logn = ring_dim.bit_length() - 1
    values = np.arange(ring_dim, dtype=np.uint32)
    values = ((values & np.uint32(0x55555555)) << np.uint32(1)) | ((values >> np.uint32(1)) & np.uint32(0x55555555))
    values = ((values & np.uint32(0x33333333)) << np.uint32(2)) | ((values >> np.uint32(2)) & np.uint32(0x33333333))
    values = ((values & np.uint32(0x0F0F0F0F)) << np.uint32(4)) | ((values >> np.uint32(4)) & np.uint32(0x0F0F0F0F))
    values = ((values & np.uint32(0x00FF00FF)) << np.uint32(8)) | ((values >> np.uint32(8)) & np.uint32(0x00FF00FF))
    values = (values << np.uint32(16)) | (values >> np.uint32(16))
    result = (values >> np.uint32(32 - logn)).astype(np.int32)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _compute_auto_map_cached(auto_index: int, ring_dim: int) -> np.ndarray:
    auto_index = int(auto_index)
    ring_dim = int(ring_dim)
    if ring_dim <= 0 or ring_dim & (ring_dim - 1):
        raise ValueError(f"ring_dim must be a positive power of two, got {ring_dim}")

    cycl_order = ring_dim << 1
    bit_reversed = bit_reverse_indices(ring_dim)
    j = np.arange(ring_dim, dtype=np.uint64)
    idx = ((((j << np.uint64(1)) + np.uint64(1)) * np.uint64(auto_index)) & np.uint64(cycl_order - 1)) >> np.uint64(1)
    result = np.zeros(ring_dim, dtype=np.int32)
    result[bit_reversed] = bit_reversed[idx.astype(np.intp)]
    result.setflags(write=False)
    return result


def compute_auto_map(auto_index, ring_dim):
    return np.array(_compute_auto_map_cached(int(auto_index), int(ring_dim)), copy=True)


def plan_rotation_groups(rotIndex_list, logBsSlots_list, levelBudget_list, logN, secretKeyDist, no_bs):
    ring_dim = 1 << int(logN)
    cycl_order = ring_dim << 1
    rotation_groups = []
    auto_idx_to_rot_idx = {}

    slot_conversion_rot_index = [1 << i for i in range(8, int(logN))]
    app_rotations = (
        list(slot_conversion_rot_index)
        if rotIndex_list is None
        else list(rotIndex_list) + slot_conversion_rot_index
    )
    if app_rotations:
        rotation_groups.append(app_rotations)
        for rot_idx in app_rotations:
            auto_idx_to_rot_idx[find_auto_index_2n_complex(rot_idx, cycl_order)] = int(rot_idx)

    if not no_bs:
        for logBsSlots, level_budget in zip(logBsSlots_list, levelBudget_list):
            bootstrap_map = bootstrap_auto_index_map(
                ring_dim,
                logBsSlots,
                level_budget,
                secretKeyDist,
            )
            if bootstrap_map:
                rotation_groups.append(list(bootstrap_map.values()))
                auto_idx_to_rot_idx.update({int(auto_idx): int(rot_idx) for auto_idx, rot_idx in bootstrap_map.items()})

            # Keep the conjugation key in a separate batch to match the key layout
            # expected by the native sampler.
            rotation_groups.append([cycl_order - 1])
            auto_idx_to_rot_idx[cycl_order - 1] = cycl_order - 1

    return rotation_groups, auto_idx_to_rot_idx


def normalize_rotation_index(rotation, ring_dim):
    rotation = int(rotation)
    if rotation < 0:
        return int((int(ring_dim) // 2) + rotation)
    return rotation


def auto_idx_to_rotation_map(rotations, ring_dim):
    cycl_order = int(ring_dim) << 1
    return {
        int(find_auto_index_2n_complex(rotation, cycl_order)): int(rotation)
        for rotation in rotations
    }


def missing_rotation_groups(crypto_context, rotation_groups):
    missing_groups = []
    for group in rotation_groups:
        missing = []
        for rotation in group:
            norm_index = normalize_rotation_index(rotation, crypto_context.N)
            if norm_index not in crypto_context.left_rot_key_map:
                missing.append(int(rotation))
        if missing:
            missing_groups.append(missing)
    return missing_groups


def install_rotation_keys(crypto_context, rotation_keys, auto_idx_to_rot_idx):
    if not rotation_keys:
        return 0

    L = int(crypto_context.L)
    K = int(crypto_context.K)
    N = int(crypto_context.N)
    dnum = int(crypto_context.dnum)
    alpha = (L + dnum - 1) // dnum
    target_device = (
        crypto_context.device
        if crypto_context.options.resolved_auto_load_keys(crypto_context.device) and crypto_context.device == "cuda"
        else "cpu"
    )
    installed = 0

    for item in rotation_keys:
        auto_idx, bx, ax = item[:3]
        auto_idx = int(auto_idx)
        if auto_idx not in auto_idx_to_rot_idx:
            raise RuntimeError(f"native sampler returned unexpected rotation auto index {auto_idx}")
        rot_idx = normalize_rotation_index(auto_idx_to_rot_idx[auto_idx], N)

        limb = crypto_context.options.rotation_key_limb_limits.get(int(rot_idx), L)
        beta = (int(limb) + alpha - 1) // alpha

        if len(item) > 3:
            trimmed_bx = np.asarray(bx, dtype=np.uint64).reshape(beta, int(limb) + K, N)
            trimmed_ax = np.asarray(ax, dtype=np.uint64).reshape(beta, int(limb) + K, N)
        else:
            reshaped_bx = np.asarray(bx, dtype=np.uint64).reshape(dnum, -1, N)
            reshaped_ax = np.asarray(ax, dtype=np.uint64).reshape(dnum, -1, N)
            if int(limb) == L and beta == dnum:
                trimmed_bx = reshaped_bx
                trimmed_ax = reshaped_ax
            else:
                trimmed_bx = np.concatenate(
                    [reshaped_bx[:beta, :limb, :], reshaped_bx[:beta, L:L + K, :]],
                    axis=1,
                )
                trimmed_ax = np.concatenate(
                    [reshaped_ax[:beta, :limb, :], reshaped_ax[:beta, L:L + K, :]],
                    axis=1,
                )

        crypto_context.left_rot_key_map[int(rot_idx)] = [
            torch.as_tensor(trimmed_bx, dtype=torch.uint64, device=target_device),
            torch.as_tensor(trimmed_ax, dtype=torch.uint64, device=target_device),
        ]
        crypto_context.precompute_auto_map[int(rot_idx)] = torch.as_tensor(
            compute_auto_map(auto_idx, N),
            dtype=torch.int32,
            device=target_device,
        )
        installed += 1

    return installed


def ensure_rotation_keys(crypto_context, rotation_groups):
    if not rotation_groups:
        return 0
    if rotation_groups and isinstance(rotation_groups[0], (int, np.integer)):
        rotation_groups = [list(rotation_groups)]
    rotation_groups = [list(group) for group in rotation_groups]
    missing_groups = missing_rotation_groups(crypto_context, rotation_groups)
    if not missing_groups:
        return 0

    sampler_config = getattr(crypto_context, "_sampler_config", None)
    if sampler_config is None:
        raise RuntimeError("Cannot generate rotation keys without native sampler config")
    key_material = crypto_context._require_key_material()
    if key_material.secret_key_coeff is None:
        raise RuntimeError("Cannot generate rotation keys without coefficient-form secret key")

    auto_map = {}
    for group in missing_groups:
        auto_map.update(auto_idx_to_rotation_map(group, crypto_context.N))

    max_limbs_by_rot = getattr(crypto_context.options, "rotation_key_limb_limits", None) or {}
    trim_by_auto = {}
    for auto_idx, rot_idx in auto_map.items():
        normalized_rot = normalize_rotation_index(rot_idx, crypto_context.N)
        if normalized_rot in max_limbs_by_rot:
            trim_by_auto[int(auto_idx)] = int(max_limbs_by_rot[normalized_rot])

    sampler_config = replace(sampler_config, rotation_trim_limbs_by_auto_index=trim_by_auto)
    provider = NativeCppSampleProvider(sampler_config)
    rotation_keys = provider.generate_rotation_keys(
        key_material.secret_key,
        key_material.secret_key_coeff,
        missing_groups,
    )
    return install_rotation_keys(crypto_context, rotation_keys, auto_map)
