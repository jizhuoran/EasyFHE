import numpy as np
import pytest

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ops import kernels, layout, rotation


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA FHE kernels required",
)


def test_u64_innerproduct_key_list_supports_64_rotation_keys():
    offsets = tuple(range(1, 65))
    slots = 128
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=2,
            log_n=14,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            rotations=offsets,
        ),
        device="cuda",
    )
    values = np.linspace(-0.75, 0.75, slots, dtype=np.float64)
    cipher = client.encrypt(values, slots=slots)

    rotated = fhe.fast_rotate(cipher, offsets, context)
    assert rotated.batch_size == len(offsets)
    for batch_id in (0, 31, 63):
        expected = fhe.homo_rotate(cipher, offsets[batch_id], context)
        np.testing.assert_allclose(
            client.decrypt(layout.cipher_batch_item(rotated, batch_id)).cpu().numpy()[:slots],
            client.decrypt(expected).cpu().numpy()[:slots],
            rtol=1e-4,
            atol=1e-4,
        )

    digits = rotation._modup_to_ext(cipher, context)
    swk_bxs, swk_axs, starts = rotation._batch_rotation_keys_and_starts(
        offsets,
        context,
        cipher.state.cur_limbs,
    )
    broadcast_bx, broadcast_ax = kernels.cv_innerproduct_broadcast(
        digits.cv[0],
        digits.state.cur_limbs,
        starts,
        swk_bxs,
        swk_axs,
        context,
    )
    pairwise_input = digits.cv[0].expand(len(offsets), -1, -1).contiguous()
    pairwise_bx, pairwise_ax = kernels.cv_innerproduct_pairwise(
        pairwise_input,
        digits.state.cur_limbs,
        starts,
        swk_bxs,
        swk_axs,
        context,
    )

    np.testing.assert_array_equal(pairwise_bx.cpu().numpy(), broadcast_bx.cpu().numpy())
    np.testing.assert_array_equal(pairwise_ax.cpu().numpy(), broadcast_ax.cpu().numpy())
