from types import SimpleNamespace

import easyfhe as torch
import numpy as np
import pytest

import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher, CipherState
from easyfhe.fhe.ops import arithmetic, kernels, rotation


def _cipher(name, cv_count=2):
    cipher = Cipher(
        [torch.zeros((2, 4), dtype=torch.uint64) for _ in range(cv_count)],
        CipherState(2, 1, 1.0),
        slots=4,
        is_ext=False,
    )
    cipher.name = name
    return cipher


def test_grouped_pairwise_mac_group_one_requires_prepacked_batches(monkeypatch):
    seen = {}

    def fake_native(ciphers, plaintexts, groups, context):
        seen["groups"] = groups
        seen["cipher_batch_size"] = ciphers.batch_size
        seen["plaintext_batch_size"] = plaintexts.batch_size
        return [torch.zeros((groups, 2, 4), dtype=torch.uint64) for _ in range(2)]

    monkeypatch.setattr(arithmetic.F, "cipher_grouped_pairwise_mac", fake_native)
    ciphers = rotation._pack_ciphers([_cipher("a"), _cipher("b")])
    plaintexts = rotation._pack_ciphers([_cipher("x", cv_count=1), _cipher("y", cv_count=1)])

    result = fhe.grouped_pairwise_mac(ciphers, plaintexts, 1, SimpleNamespace())

    assert seen == {
        "groups": 1,
        "cipher_batch_size": 2,
        "plaintext_batch_size": 2,
    }
    assert result.batch_size == 1
    assert tuple(result.cv[0].shape) == (1, 2, 4)


def test_grouped_pairwise_mac_reuses_cipher_batch(monkeypatch):
    seen = {}

    def fake_native(ciphers, plaintexts, groups, context):
        seen["groups"] = groups
        seen["cipher_batch_size"] = ciphers.batch_size
        seen["plaintext_batch_size"] = plaintexts.batch_size
        return [torch.zeros((groups, 2, 4), dtype=torch.uint64) for _ in range(2)]

    monkeypatch.setattr(arithmetic.F, "cipher_grouped_pairwise_mac", fake_native)
    ciphers = rotation._pack_ciphers([_cipher("a"), _cipher("b")])
    plaintexts = rotation._pack_ciphers([
        _cipher("x0", cv_count=1),
        _cipher("y0", cv_count=1),
        _cipher("x1", cv_count=1),
        _cipher("y1", cv_count=1),
    ])

    result = fhe.grouped_pairwise_mac(ciphers, plaintexts, 2, SimpleNamespace())

    assert seen == {
        "groups": 2,
        "cipher_batch_size": 2,
        "plaintext_batch_size": 4,
    }
    assert result.batch_size == 2
    assert tuple(result.cv[0].shape) == (2, 2, 4)


def test_grouped_pairwise_mac_rejects_lists():
    with pytest.raises(TypeError, match="expected a batched Cipher"):
        fhe.grouped_pairwise_mac(
            [_cipher("a"), _cipher("b")],
            [_cipher("x", cv_count=1), _cipher("y", cv_count=1)],
            1,
            SimpleNamespace(),
        )


def test_grouped_pairwise_mac_group_one_rejects_mismatched_batch_lengths():
    with pytest.raises(ValueError, match="plaintext batch size must equal groups"):
        fhe.grouped_pairwise_mac(
            rotation._pack_ciphers([_cipher("a"), _cipher("b")]),
            rotation._pack_ciphers([_cipher("x", cv_count=1)]),
            1,
            SimpleNamespace(),
        )


def test_grouped_pairwise_mac_rejects_mismatched_grouped_batch_lengths():
    with pytest.raises(ValueError, match="plaintext batch size must equal groups"):
        fhe.grouped_pairwise_mac(
            rotation._pack_ciphers([_cipher("a"), _cipher("b")]),
            rotation._pack_ciphers([
                _cipher("x0", cv_count=1),
                _cipher("y0", cv_count=1),
                _cipher("x1", cv_count=1),
            ]),
            2,
            SimpleNamespace(),
        )


def test_grouped_scalar_weighted_acc_matches_scalar_mul_add_loop():
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=6, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    batch_size = 4
    cur_limbs = 3
    rng = np.random.default_rng(0)
    cv = []
    for _ in range(2):
        rows = []
        for _ in range(batch_size):
            rows.append(
                np.stack([
                    rng.integers(0, int(ctx.moduliQ_scalar[level]), size=ctx.N, dtype=np.uint64)
                    for level in range(cur_limbs)
                ])
            )
        cv.append(torch.from_numpy(np.stack(rows, axis=0)))

    scalars = torch.from_numpy(
        np.stack([
            np.asarray([
                rng.integers(0, int(ctx.moduliQ_scalar[level]), dtype=np.uint64)
                for level in range(cur_limbs)
            ], dtype=np.uint64)
            for _ in range(batch_size)
        ])
    )
    cipher = Cipher(
        cv,
        CipherState(cur_limbs, 1, 1.0),
        slots=ctx.N // 2,
        is_ext=False,
        batch_size=batch_size,
    )

    actual = fhe.grouped_scalar_weighted_acc(cipher, scalars.unsqueeze(0), ctx)
    expected = []
    for component in cipher.cv:
        acc = None
        for index in range(batch_size):
            term = kernels.cv_mul_scalar(
                component[index : index + 1],
                scalars[index],
                ctx.moduliQ,
                ctx.q_mu,
                cur_limbs,
            )
            acc = term if acc is None else kernels.cv_add(acc, term, ctx.moduliQ, cur_limbs)
        expected.append(acc)

    assert actual.batch_size == 1
    assert actual.state == CipherState(cur_limbs, 2, ctx.scale_at(cur_limbs))
    assert all(np.array_equal(a.numpy(), e.numpy()) for a, e in zip(actual.cv, expected))
