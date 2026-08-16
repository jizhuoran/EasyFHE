import numpy as np
import pytest

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ops import layout, rotation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="batched fast rotation uses CUDA-only native kernels",
)


def _plaintext(values, context, *, level, slots, is_ext=False):
    vector = fhe.PackedRaw(torch.from_numpy(np.asarray(values)))
    cur_limbs = context.L - int(level)
    return fhe.ConstantBundle(vectors={"pt": vector}, cache_mode="none").plaintext(
        "pt",
        state=fhe.CipherState(cur_limbs, 1, context.scale_at(cur_limbs)),
        slots=slots,
        context=context,
        is_ext=is_ext,
    )


def _client_context(spec, device):
    return fhe.generate_client_context(spec, device=device)


def _mac_down_each(cipher, offsets, plain_values, context):
    rotated = fhe.fast_rotate(cipher, offsets, context)
    total = None
    for index, values in enumerate(plain_values):
        rot = layout.cipher_batch_item(rotated, index)
        plaintext = _plaintext(
            values,
            context,
            level=context.L - rot.state.cur_limbs,
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
            level=context.L - rotated_ext.state.cur_limbs,
            slots=rotated_ext.slots,
            is_ext=True,
        )
        plaintexts.append(plaintext_ext)
    total_ext = fhe.grouped_pairwise_mac(rotated_ext, layout.pack_cipher_batch(plaintexts), 1, context)
    return fhe.moddown_from_ext(layout.cipher_batch_item(total_ext, 0), context)


def _hoisted_mac(cipher, offsets, plain_values, context):
    plaintexts = []
    for values in plain_values:
        plaintext_ext = _plaintext(
            values,
            context,
            level=context.L - cipher.state.cur_limbs,
            slots=cipher.slots,
            is_ext=True,
        )
        plaintexts.append(plaintext_ext)
    return fhe.hoisted_mac_sum(
        cipher,
        offsets,
        layout.pack_cipher_batch(plaintexts),
        0,
        1,
        context,
        strategy="ext_double_hoist",
    )


def _hoisted_mac_ext_double(cipher, offsets, groups, giant_offset, context):
    plaintexts = []
    for group in groups:
        for values in group:
            plaintexts.append(
                _plaintext(
                    values,
                    context,
                    level=context.L - cipher.state.cur_limbs,
                    slots=cipher.slots,
                    is_ext=True,
                )
            )
    return fhe.hoisted_mac_sum(
        cipher,
        offsets,
        layout.pack_cipher_batch(plaintexts),
        giant_offset,
        len(groups),
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
                    level=context.L - cipher.state.cur_limbs,
                    slots=cipher.slots,
                )
            )
    return fhe.hoisted_mac_sum(
        cipher,
        offsets,
        layout.pack_cipher_batch(plaintexts),
        giant_offset,
        len(groups),
        context,
        strategy="normal",
    )


def _grouped_plaintexts(cipher, groups, context, *, is_ext):
    plaintexts = []
    for group in groups:
        for values in group:
            plaintexts.append(
                _plaintext(
                    values,
                    context,
                    level=context.L - cipher.state.cur_limbs,
                    slots=cipher.slots,
                    is_ext=bool(is_ext),
                )
            )
    return layout.pack_cipher_batch(plaintexts)


def _manual_normal_grouped_mac(cipher, offsets, groups, giant_offset, context):
    rotations = fhe.fast_rotate(cipher, offsets, context)
    partial_sums = []
    for group in groups:
        total = None
        for index, values in enumerate(group):
            rotated = layout.cipher_batch_item(rotations, index)
            plaintext = _plaintext(
                values,
                context,
                level=context.L - rotated.state.cur_limbs,
                slots=rotated.slots,
            )
            term = fhe.homo_mul_pt(rotated, plaintext, context)
            total = term if total is None else fhe.homo_add(total, term, context)
        partial_sums.append(total)
    return fhe.giant_rotate_sum(
        layout.pack_cipher_batch(partial_sums), giant_offset, context, strategy="normal"
    )


def test_fast_rotate_ext_mac_can_defer_moddown_once():
    slots = 1024
    offsets = list(range(16))
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=tuple(offset for offset in offsets if offset),
        ),
        "cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)
    plaintext_values = [
        np.full(slots, 0.01 * (idx + 1), dtype=np.double)
        for idx in range(len(offsets))
    ]

    down_each = _mac_down_each(cipher, offsets, plaintext_values, context)
    down_once = _mac_down_once(cipher, offsets, plaintext_values, context)
    hoisted = _hoisted_mac(cipher, offsets, plaintext_values, context)

    assert down_each.state.cur_limbs == down_once.state.cur_limbs
    assert down_each.is_ext is False
    assert down_once.is_ext is False
    assert hoisted.state.cur_limbs == down_once.state.cur_limbs
    assert hoisted.is_ext is False
    np.testing.assert_allclose(
        client.decrypt(down_each).cpu().numpy()[:slots],
        client.decrypt(down_once).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        client.decrypt(down_once).cpu().numpy()[:slots],
        client.decrypt(hoisted).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )


def test_hoisted_mac_sum_normal_matches_manual_grouped_path():
    slots = 1024
    baby_offsets = [0, 1, 2]
    giant_offset = 5
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=(1, 2, giant_offset, 2 * giant_offset),
        ),
        "cuda",
    )
    values = np.linspace(-0.5, 0.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)
    groups = [
        [np.full(slots, 0.01 * (group + 1) * (idx + 1), dtype=np.double) for idx in range(len(baby_offsets))]
        for group in range(3)
    ]

    manual = _manual_normal_grouped_mac(cipher, baby_offsets, groups, giant_offset, context)
    hoisted = _hoisted_mac_normal(cipher, baby_offsets, groups, giant_offset, context)

    np.testing.assert_allclose(
        client.decrypt(manual).cpu().numpy()[:slots],
        client.decrypt(hoisted).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )


def test_hoisted_mac_sum_can_return_and_reuse_baby_rotations():
    slots = 1024
    baby_offsets = [0, 1, 2, 3, 4, 5]
    giant_offset = 7
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            # max_fast_steps=2 needs only local key 1 and anchor key 2.
            # Direct fast rotation of all six offsets would also need 3,4,5.
            rotations=(1, 2, giant_offset, 2 * giant_offset),
        ),
        "cuda",
    )
    values = np.linspace(-0.5, 0.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)
    groups = [
        [
            np.full(
                slots,
                0.01 * (group + 1) * (idx + 1),
                dtype=np.double,
            )
            for idx in range(len(baby_offsets))
        ]
        for group in range(3)
    ]
    plaintexts = _grouped_plaintexts(
        cipher, groups, context, is_ext=True
    )

    first, baby_rotations = fhe.hoisted_mac_sum(
        cipher,
        baby_offsets,
        plaintexts,
        giant_offset,
        len(groups),
        context,
        strategy="ext_double_hoist",
        baby_anchor_step=2,
        return_baby_rotations=True,
    )
    second, returned_baby_rotations = fhe.hoisted_mac_sum(
        cipher,
        baby_offsets,
        plaintexts,
        giant_offset,
        len(groups),
        context,
        strategy="ext_double_hoist",
        baby_anchor_step=2,
        baby_rotations=baby_rotations,
        return_baby_rotations=True,
    )
    explicitly_prepared = fhe.prepare_hoisted_baby_rotations(
        cipher,
        baby_offsets,
        context,
        strategy="ext_double_hoist",
        baby_anchor_step=2,
    )
    third = fhe.hoisted_mac_sum(
        cipher,
        baby_offsets,
        plaintexts,
        giant_offset,
        len(groups),
        context,
        strategy="ext_double_hoist",
        baby_anchor_step=2,
        baby_rotations=explicitly_prepared,
    )

    assert returned_baby_rotations is baby_rotations
    assert tuple(block.batch_size for block in baby_rotations) == (2, 2, 2)
    np.testing.assert_allclose(
        client.decrypt(first).cpu().numpy()[:slots],
        client.decrypt(second).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        client.decrypt(first).cpu().numpy()[:slots],
        client.decrypt(third).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )


def test_hoisted_mac_sum_ext_double_uses_giant_step_offsets():
    slots = 1024
    baby_offsets = [0, 1, 2]
    giant_offset = 5
    group_count = 3
    giant_keys = [giant_offset * index for index in range(1, group_count)]
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=5,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=tuple(sorted({*baby_offsets[1:], *giant_keys})),
        ),
        "cuda",
    )
    values = np.linspace(-0.75, 0.75, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)
    groups = [
        [np.full(slots, 0.01 * (group + 1) * (idx + 1), dtype=np.double) for idx in range(len(baby_offsets))]
        for group in range(group_count)
    ]

    manual = _manual_normal_grouped_mac(cipher, baby_offsets, groups, giant_offset, context)
    hoisted = _hoisted_mac_ext_double(cipher, baby_offsets, groups, giant_offset, context)

    np.testing.assert_allclose(
        client.decrypt(manual).cpu().numpy()[:slots],
        client.decrypt(hoisted).cpu().numpy()[:slots],
        rtol=1e-4,
        atol=1e-4,
    )


def test_fast_rotate_shapes_match_offsets():
    slots = 1024
    offsets = [0, 1, 2, 3]
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=tuple(offset for offset in offsets if offset),
        ),
        "cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)

    normal_batch = fhe.fast_rotate(cipher, offsets, context)
    ext_batch = fhe.fast_rotate(cipher, offsets, context, output_ext=True)

    assert normal_batch.batch_size == len(offsets)
    assert ext_batch.batch_size == len(offsets)
    assert normal_batch.cv[0].shape[0] == len(offsets)
    assert ext_batch.cv[0].shape[0] == len(offsets)
    assert normal_batch.is_ext is False
    assert ext_batch.is_ext is True


def test_fast_rotate_rejects_all_zero_offsets():
    slots = 1024
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=(),
        ),
        "cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)

    with pytest.raises(ValueError, match="at least one nonzero offset"):
        fhe.fast_rotate(cipher, [0, 0], context)


def test_giant_rotate_sum_rejects_zero_offset():
    slots = 1024
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=(),
        ),
        "cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)

    with pytest.raises(ValueError, match="offset must be nonzero"):
        fhe.giant_rotate_sum((cipher,), 0, context, strategy="normal")


def test_giant_rotate_sum_requires_batched_cipher():
    slots = 1024
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=(1,),
        ),
        "cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)

    with pytest.raises(TypeError, match="expected a batched Cipher"):
        fhe.giant_rotate_sum((cipher,), 1, context, strategy="normal")


def test_giant_rotate_sum_requires_multiple_cipher_batch():
    slots = 1024
    client, context = _client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=(1,),
        ),
        "cuda",
    )
    values = np.linspace(0.0, 1.5, slots, dtype=np.double)
    cipher = client.encrypt(values, slots=slots)
    batched = cipher.cipher_like(cipher.cv, batch_size=1)

    with pytest.raises(ValueError, match="batch_size > 1"):
        fhe.giant_rotate_sum(batched, 1, context, strategy="normal")
