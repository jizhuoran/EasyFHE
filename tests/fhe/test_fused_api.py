from types import SimpleNamespace

import easyfhe as torch
import numpy as np
import pytest

import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher, CipherState
from easyfhe.fhe.ops import arithmetic, kernels, layout


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
    ciphers = layout.pack_cipher_batch([_cipher("a"), _cipher("b")])
    plaintexts = layout.pack_cipher_batch([_cipher("x", cv_count=1), _cipher("y", cv_count=1)])

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
    ciphers = layout.pack_cipher_batch([_cipher("a"), _cipher("b")])
    plaintexts = layout.pack_cipher_batch([
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
            layout.pack_cipher_batch([_cipher("a"), _cipher("b")]),
            layout.pack_cipher_batch([_cipher("x", cv_count=1)]),
            1,
            SimpleNamespace(),
        )


def test_grouped_pairwise_mac_rejects_mismatched_grouped_batch_lengths():
    with pytest.raises(ValueError, match="plaintext batch size must equal groups"):
        fhe.grouped_pairwise_mac(
            layout.pack_cipher_batch([_cipher("a"), _cipher("b")]),
            layout.pack_cipher_batch([
                _cipher("x0", cv_count=1),
                _cipher("y0", cv_count=1),
                _cipher("x1", cv_count=1),
            ]),
            2,
            SimpleNamespace(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pairwise MAC requires CUDA")
def test_batched_pairwise_mac_32x64_cuda_matches_direct_and_numpy(monkeypatch):
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=2, log_n=10, dnum=1, dcrt_bits=30, first_mod=35),
        device="cuda",
    )
    num_batches = 32
    num_cipher = 64
    cur_limbs = 2
    N = ctx.N
    rng = np.random.default_rng(1)

    cipher_bx_np = rng.integers(0, 17, size=(num_cipher, cur_limbs, N), dtype=np.uint64)
    cipher_ax_np = rng.integers(0, 17, size=(num_cipher, cur_limbs, N), dtype=np.uint64)
    plaintext_np = rng.integers(
        0,
        17,
        size=(num_batches * num_cipher, cur_limbs, N),
        dtype=np.uint64,
    )
    primes_np = ctx.QplusP_map[cur_limbs].cpu().numpy()

    plain_grouped = plaintext_np.reshape(num_batches, num_cipher, cur_limbs, N)
    expected_bx = np.empty((num_batches, cur_limbs, N), dtype=np.uint64)
    expected_ax = np.empty_like(expected_bx)
    for limb in range(cur_limbs):
        prime = np.uint64(primes_np[limb])
        expected_bx[:, limb, :] = (
            (plain_grouped[:, :, limb, :] * cipher_bx_np[None, :, limb, :]).sum(
                axis=1,
                dtype=np.uint64,
            )
            % prime
        )
        expected_ax[:, limb, :] = (
            (plain_grouped[:, :, limb, :] * cipher_ax_np[None, :, limb, :]).sum(
                axis=1,
                dtype=np.uint64,
            )
            % prime
        )

    cipher_bx = torch.from_numpy(cipher_bx_np).to("cuda")
    cipher_ax = torch.from_numpy(cipher_ax_np).to("cuda")
    plaintext = torch.from_numpy(plaintext_np).to("cuda")

    monkeypatch.delenv("EASYFHE_PAIRWISE_MAC_32_64_DIRECT", raising=False)
    shared_bx, shared_ax = torch.batched_pairwise_mac(
        cipher_bx,
        cipher_ax,
        plaintext,
        ctx.QplusP_map[cur_limbs],
        ctx.QbarretRatioplusPbarretRatio_map[cur_limbs],
        ctx.QbarretKplusPbarretK_map[cur_limbs],
        num_batches,
        num_cipher,
        cur_limbs,
        N,
    )
    torch.cuda.synchronize()

    monkeypatch.setenv("EASYFHE_PAIRWISE_MAC_32_64_DIRECT", "1")
    direct_bx, direct_ax = torch.batched_pairwise_mac(
        cipher_bx,
        cipher_ax,
        plaintext,
        ctx.QplusP_map[cur_limbs],
        ctx.QbarretRatioplusPbarretRatio_map[cur_limbs],
        ctx.QbarretKplusPbarretK_map[cur_limbs],
        num_batches,
        num_cipher,
        cur_limbs,
        N,
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(shared_bx.cpu().numpy(), expected_bx)
    np.testing.assert_array_equal(shared_ax.cpu().numpy(), expected_ax)
    np.testing.assert_array_equal(direct_bx.cpu().numpy(), expected_bx)
    np.testing.assert_array_equal(direct_ax.cpu().numpy(), expected_ax)


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

    encoded = fhe.EncodedScalar(
        scalars.unsqueeze(0),
        cur_limbs=cur_limbs,
        scale_degree=1,
        scaling_factor=ctx.scale_at(cur_limbs),
    )
    actual = fhe.grouped_scalar_weighted_acc(cipher, encoded, ctx)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped accumulation requires CUDA")
@pytest.mark.parametrize("num_groups", [6, 7])
def test_u64_grouped_scalar_weighted_acc_resnet_shapes_cuda(num_groups):
    _, cpu_ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=2, log_n=6, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    ctx = cpu_ctx.cuda()
    num_cipher = 6
    cur_limbs = 2
    rng = np.random.default_rng(num_groups)
    bx_np = rng.integers(0, 17, size=(num_cipher, cur_limbs, ctx.N), dtype=np.uint64)
    ax_np = rng.integers(0, 17, size=(num_cipher, cur_limbs, ctx.N), dtype=np.uint64)
    scalars_np = rng.integers(0, 17, size=(num_groups, num_cipher, cur_limbs), dtype=np.uint64)

    actual_bx, actual_ax = torch.grouped_scalar_weighted_acc(
        torch.from_numpy(bx_np).cuda(),
        torch.from_numpy(ax_np).cuda(),
        torch.from_numpy(scalars_np).cuda(),
        ctx.moduliQ,
        ctx.QbarretRatioplusPbarretRatio_map[cur_limbs],
        ctx.QbarretKplusPbarretK_map[cur_limbs],
        num_groups,
        num_cipher,
        cur_limbs,
        ctx.N,
        -1,
    )
    torch.cuda.synchronize()

    expected_bx = np.empty((num_groups, cur_limbs, ctx.N), dtype=np.uint64)
    expected_ax = np.empty_like(expected_bx)
    for group in range(num_groups):
        for limb in range(cur_limbs):
            modulus = np.uint64(ctx.moduliQ_scalar[limb])
            weights = scalars_np[group, :, limb, None]
            expected_bx[group, limb] = np.sum(bx_np[:, limb] * weights, axis=0) % modulus
            expected_ax[group, limb] = np.sum(ax_np[:, limb] * weights, axis=0) % modulus

    np.testing.assert_array_equal(actual_bx.cpu().numpy(), expected_bx)
    np.testing.assert_array_equal(actual_ax.cpu().numpy(), expected_ax)
