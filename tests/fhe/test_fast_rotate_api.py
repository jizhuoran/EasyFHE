import numpy as np

import easyfhe.fhe as fhe
from easyfhe.fhe.ops import rotation


def _mac_down_each(cipher, offsets, plain_values, context):
    rotated = fhe.fast_rotate(cipher, offsets, context)
    total = None
    for rot, values in zip(rotated, plain_values):
        plaintext = fhe.encode(
            values,
            context,
            level=context.L - rot.cur_limbs,
            slots=rot.slots,
        )[1]
        term = fhe.homo_mul_pt(rot, plaintext, context)
        total = term if total is None else fhe.homo_add(total, term, context)
    return total


def _mac_down_once(cipher, offsets, plain_values, context):
    rotated_ext = fhe.fast_rotate_ext_batch(cipher, offsets, context)
    plaintexts = []
    for values in plain_values:
        plaintext_ext = fhe.encode(
            values,
            context,
            level=context.L - rotated_ext.cur_limbs,
            slots=rotated_ext.slots,
            is_ext=True,
        )[1]
        plaintexts.append(plaintext_ext)
    total_ext = fhe.fused_pairwise_mac(rotated_ext, rotation._pack_ciphers(plaintexts), context)
    return fhe.double_hoist_rotate_sum([total_ext], [0], context)


def test_fast_rotate_ext_mac_can_defer_moddown_once():
    slots = 16
    offsets = list(range(16))
    context = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=6,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=tuple(offset for offset in offsets if offset),
        ),
        device="cpu",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = context.encrypt(values, "cpu", 1, 0, slots)
    plaintext_values = [
        np.full(slots, 0.01 * (idx + 1), dtype=np.double)
        for idx in range(len(offsets))
    ]

    down_each = _mac_down_each(cipher, offsets, plaintext_values, context)
    down_once = _mac_down_once(cipher, offsets, plaintext_values, context)

    assert down_each.cur_limbs == down_once.cur_limbs
    assert down_each.is_ext is False
    assert down_once.is_ext is False
    np.testing.assert_allclose(
        context.decrypt(down_each).cpu().numpy()[:slots],
        context.decrypt(down_once).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )


def test_fast_rotate_batch_shapes_match_offsets():
    slots = 16
    offsets = [0, 1, 2, 3]
    context = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=6,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=tuple(offset for offset in offsets if offset),
        ),
        device="cpu",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = context.encrypt(values, "cpu", 1, 0, slots)

    normal_batch = fhe.fast_rotate_batch(cipher, offsets, context)
    ext_batch = fhe.fast_rotate_ext_batch(cipher, offsets, context)

    assert normal_batch.batch_size == len(offsets)
    assert ext_batch.batch_size == len(offsets)
    assert normal_batch.cv[0].shape[0] == len(offsets)
    assert ext_batch.cv[0].shape[0] == len(offsets)
    assert normal_batch.is_ext is False
    assert ext_batch.is_ext is True
