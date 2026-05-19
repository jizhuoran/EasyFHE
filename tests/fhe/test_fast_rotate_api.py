import numpy as np
import pytest

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ops import rotation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="batched fast rotation uses CUDA-only native kernels",
)


def _plaintext(values, context, *, level, slots, is_ext=False):
    return fhe.ConstantBundle(vectors={"pt": values}, cache_mode="none").plaintext(
        "pt",
        level,
        slots,
        context,
        is_ext=is_ext,
    )


def _mac_down_each(cipher, offsets, plain_values, context):
    rotated = fhe.fast_rotate(cipher, offsets, context)
    total = None
    for index, values in enumerate(plain_values):
        rot = rotation._batch_item(rotated, index)
        plaintext = _plaintext(
            values,
            context,
            level=context.L - rot.cur_limbs,
            slots=rot.slots,
        )
        term = fhe.homo_mul_pt(rot, plaintext, context)
        total = term if total is None else fhe.homo_add(total, term, context)
    return total


def _mac_down_once(cipher, offsets, plain_values, context):
    rotated_ext = fhe.fast_rotate(cipher, offsets, context, output_ext=True)
    plaintexts = []
    for values in plain_values:
        plaintext_ext = _plaintext(
            values,
            context,
            level=context.L - rotated_ext.cur_limbs,
            slots=rotated_ext.slots,
            is_ext=True,
        )
        plaintexts.append(plaintext_ext)
    total_ext = fhe.fused_grouped_pairwise_mac(rotated_ext, rotation._pack_ciphers(plaintexts), 1, context)[0]
    return fhe.giant_rotate_sum([total_ext], 0, context, strategy="ext_double_hoist")


def _hoisted_mac(cipher, offsets, plain_values, context):
    plaintexts = []
    for values in plain_values:
        plaintext_ext = _plaintext(
            values,
            context,
            level=context.L - cipher.cur_limbs,
            slots=cipher.slots,
            is_ext=True,
        )
        plaintexts.append(plaintext_ext)
    return fhe.hoisted_mac_sum(
        cipher,
        offsets,
        rotation._pack_ciphers(plaintexts),
        0,
        1,
        context,
        strategy="ext_double_hoist",
    )


def _hoisted_mac_normal(cipher, offsets, groups, giant_offset, context):
    plaintexts = []
    for group in groups:
        for values in group:
            plaintexts.append(
                _plaintext(
                    values,
                    context,
                    level=context.L - cipher.cur_limbs,
                    slots=cipher.slots,
                )
            )
    return fhe.hoisted_mac_sum(
        cipher,
        offsets,
        rotation._pack_ciphers(plaintexts),
        giant_offset,
        len(groups),
        context,
        strategy="normal",
    )


def _manual_normal_grouped_mac(cipher, offsets, groups, giant_offset, context):
    rotations = fhe.fast_rotate(cipher, offsets, context)
    partial_sums = []
    for group in groups:
        total = None
        for index, values in enumerate(group):
            rotated = rotation._batch_item(rotations, index)
            plaintext = _plaintext(
                values,
                context,
                level=context.L - rotated.cur_limbs,
                slots=rotated.slots,
            )
            term = fhe.homo_mul_pt(rotated, plaintext, context)
            total = term if total is None else fhe.homo_add(total, term, context)
        partial_sums.append(total)
    return fhe.giant_rotate_sum(partial_sums, giant_offset, context, strategy="normal")


def test_fast_rotate_ext_mac_can_defer_moddown_once():
    slots = 1024
    offsets = list(range(16))
    context = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=tuple(offset for offset in offsets if offset),
        ),
        device="cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = context.encrypt(values, "cuda", 1, 0, slots)
    plaintext_values = [
        np.full(slots, 0.01 * (idx + 1), dtype=np.double)
        for idx in range(len(offsets))
    ]

    down_each = _mac_down_each(cipher, offsets, plaintext_values, context)
    down_once = _mac_down_once(cipher, offsets, plaintext_values, context)
    hoisted = _hoisted_mac(cipher, offsets, plaintext_values, context)

    assert down_each.cur_limbs == down_once.cur_limbs
    assert down_each.is_ext is False
    assert down_once.is_ext is False
    assert hoisted.cur_limbs == down_once.cur_limbs
    assert hoisted.is_ext is False
    np.testing.assert_allclose(
        context.decrypt(down_each).cpu().numpy()[:slots],
        context.decrypt(down_once).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        context.decrypt(down_once).cpu().numpy()[:slots],
        context.decrypt(hoisted).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )


def test_hoisted_mac_sum_normal_matches_manual_grouped_path():
    slots = 1024
    baby_offsets = [0, 1, 2]
    giant_offset = 5
    context = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=(1, 2, giant_offset),
        ),
        device="cuda",
    )
    values = np.linspace(-0.5, 0.5, slots, dtype=np.double)
    cipher = context.encrypt(values, "cuda", 1, 0, slots)
    groups = [
        [np.full(slots, 0.01 * (group + 1) * (idx + 1), dtype=np.double) for idx in range(len(baby_offsets))]
        for group in range(3)
    ]

    manual = _manual_normal_grouped_mac(cipher, baby_offsets, groups, giant_offset, context)
    hoisted = _hoisted_mac_normal(cipher, baby_offsets, groups, giant_offset, context)

    np.testing.assert_allclose(
        context.decrypt(manual).cpu().numpy()[:slots],
        context.decrypt(hoisted).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )


def test_fast_rotate_shapes_match_offsets():
    slots = 1024
    offsets = [0, 1, 2, 3]
    context = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=tuple(offset for offset in offsets if offset),
        ),
        device="cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = context.encrypt(values, "cuda", 1, 0, slots)

    normal_batch = fhe.fast_rotate(cipher, offsets, context)
    ext_batch = fhe.fast_rotate(cipher, offsets, context, output_ext=True)

    assert normal_batch.batch_size == len(offsets)
    assert ext_batch.batch_size == len(offsets)
    assert normal_batch.cv[0].shape[0] == len(offsets)
    assert ext_batch.cv[0].shape[0] == len(offsets)
    assert normal_batch.is_ext is False
    assert ext_batch.is_ext is True
