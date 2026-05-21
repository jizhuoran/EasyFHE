import numpy as np
from types import SimpleNamespace

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher
from easyfhe.fhe.ops.encoding import encode_stage1
from examples.resnet20_aespa.weight_pack import WeightPack


def _cipher(name, cv_count=2):
    cipher = Cipher(
        [torch.zeros((2, 4), dtype=torch.uint64) for _ in range(cv_count)],
        cur_limbs=2,
        scaling_factor=1.0,
        noise_deg=1,
        slots=4,
        is_ext=False,
    )
    cipher.name = name
    return cipher


def test_constant_bundle_encodes_middle_and_caches_vectors(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={"w": np.asarray([1.0, 2.0], dtype=np.double)},
        cache_mode="middle",
    )
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        calls.append(middle)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    first = bundle.plaintext("w", 3, 4, ctx)
    second = bundle.plaintext("w", 3, 4, ctx)

    assert first is not second
    assert calls[0] is calls[1]
    assert calls[0].slots == 4
    assert bundle.cache_info()["middle_hits"] == 1
    assert bundle.cache_info()["middle_misses"] == 1


def test_constant_bundle_accepts_prepared_vectors(monkeypatch):
    prepared = encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=2, ring_dim=8)
    bundle = fhe.ConstantBundle(vectors={"w": prepared}, cache_mode="middle")
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        calls.append(middle)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    bundle.plaintext("w", 3, 2, ctx)

    assert calls == [prepared]


def test_constant_bundle_preserves_complex_raw_values(monkeypatch):
    values = np.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex128)
    bundle = fhe.ConstantBundle(vectors={"w": values}, cache_mode="none")
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        calls.append(middle)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    bundle.plaintext("w", 3, 4, ctx)

    padded = calls[0].values
    assert padded.dtype == np.complex128
    assert np.array_equal(padded, np.asarray([1.0 + 2.0j, 3.0 - 4.0j, 0.0, 0.0]))


def test_constant_bundle_rejects_ragged_vector_batches():
    bundle = fhe.ConstantBundle(
        vectors={"w": [np.asarray([1.0]), np.asarray([2.0, 3.0])]},
        cache_mode="none",
    )

    try:
        bundle.plaintext("w", 3, 4, SimpleNamespace(N=8))
    except TypeError as exc:
        assert "rectangular numeric arrays" in str(exc)
    else:
        raise AssertionError("expected ragged vector batch to fail")


def test_constant_bundle_rejects_list_of_names():
    bundle = fhe.ConstantBundle(
        vectors={
            "a": np.asarray([1.0], dtype=np.double),
            "b": np.asarray([2.0], dtype=np.double),
        },
        cache_mode="plain",
    )

    ctx = SimpleNamespace(N=8)
    try:
        bundle.plaintext(["a", "b"], 3, 4, ctx, is_ext=True)
    except TypeError as exc:
        assert "name must be str" in str(exc)
    else:
        raise AssertionError("expected list-of-name plaintext lookup to fail")


def test_constant_bundle_named_raw_batch_matches_individual_encoding():
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    bundle = fhe.ConstantBundle(
        vectors={
            "group": np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=np.double,
            ),
        },
        cache_mode="none",
    )

    batch = bundle.plaintext("group", 1, 4, ctx, is_ext=True)
    first = fhe.ConstantBundle(vectors={"a": np.asarray([1.0, 2.0], dtype=np.double)}, cache_mode="none").plaintext(
        "a", 1, 4, ctx, is_ext=True
    )
    second = fhe.ConstantBundle(vectors={"b": np.asarray([3.0, 4.0], dtype=np.double)}, cache_mode="none").plaintext(
        "b", 1, 4, ctx, is_ext=True
    )

    assert batch.batch_size == 2
    assert batch.is_ext is True
    assert np.array_equal(batch.cv[0][0].numpy(), first.cv[0][0].numpy())
    assert np.array_equal(batch.cv[0][1].numpy(), second.cv[0][0].numpy())


def test_constant_bundle_named_raw_batch_cache(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={
            "group": np.asarray(
                [
                    [1.0],
                    [2.0],
                ],
                dtype=np.double,
            ),
        },
        cache_mode="plain",
    )
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        calls.append((middle.encoded_values.shape, level, slots, is_ext))
        return _cipher("group", cv_count=1).cipher_like([torch.zeros((2, 2, 4), dtype=torch.uint64)], batch_size=2)

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)

    ctx = SimpleNamespace(N=8)
    first = bundle.plaintext("group", 3, 4, ctx, is_ext=True)
    second = bundle.plaintext("group", 3, 4, ctx, is_ext=True)

    assert first is second
    assert first.batch_size == 2
    assert calls == [((2, 8), 3, 4, True)]
    assert bundle.cache_info()["plain_hits"] == 1
    assert bundle.cache_info()["plain_misses"] == 1


def test_constant_bundle_encodes_and_caches_scalars():
    ctx = SimpleNamespace(
        device="cpu",
        N=8,
        L=4,
        scale_mode="fixed",
        rescale_policy="manual",
        moduliQ_scalar=[17, 19, 23, 29],
        scale_at=lambda cur_limbs: 8.0,
    )
    bundle = fhe.ConstantBundle(scalars={"alpha": 1.5, "shift": -2}, cache_mode="plain")

    first = bundle.encoded_scalars("alpha", 3, 1, ctx, mode="double")
    second = bundle.encoded_scalars("alpha", 3, 1, ctx, mode="double")
    integer = bundle.encoded_scalars("shift", 2, 0, ctx, mode="int")

    assert first is second
    assert first.tolist() == [[12, 12, 12]]
    assert integer.tolist() == [[15, 17]]
    assert bundle.cache_info()["scalar_hits"] == 1
    assert bundle.cache_info()["scalar_misses"] == 2
    assert bundle.cache_info()["scalar_entries"] == 2


def test_constant_bundle_encodes_and_caches_scalar_batches():
    ctx = SimpleNamespace(
        device="cpu",
        N=8,
        L=4,
        scale_mode="fixed",
        rescale_policy="manual",
        moduliQ_scalar=[17, 19, 23, 29],
        scale_at=lambda cur_limbs: 8.0,
    )
    bundle = fhe.ConstantBundle(scalars={"a": 1.5, "b": -2.0}, cache_mode="plain")

    first = bundle.encoded_scalars(("a", "b"), 3, 1, ctx, mode="double")
    second = bundle.encoded_scalars(("a", "b"), 3, 1, ctx, mode="double")

    assert first is second
    assert first.shape == (2, 3)
    assert first.tolist() == [[12, 12, 12], [2, 4, 8]]
    assert bundle.cache_info()["scalar_hits"] == 1
    assert bundle.cache_info()["scalar_misses"] == 1
    assert bundle.cache_info()["scalar_entries"] == 1


def test_constant_bundle_encodes_double_scalars_at_requested_noise_degree():
    ctx = SimpleNamespace(
        device="cpu",
        N=8,
        L=4,
        scale_mode="fixed",
        rescale_policy="manual",
        moduliQ_scalar=[17, 19, 23, 29],
        scale_at=lambda cur_limbs: 8.0,
    )
    bundle = fhe.ConstantBundle(scalars={"shift": 1.5}, cache_mode="none")

    encoded = bundle.encoded_scalars("shift", 3, 2, ctx, mode="double")

    assert encoded.shape == (1, 3)
    assert encoded.tolist() == [[11, 1, 4]]


def test_resnet_weight_pack_reuses_constant_bundle():
    weights = WeightPack({"w": np.asarray([1.0], dtype=np.double)}, cache_mode="middle")

    assert isinstance(weights, fhe.ConstantBundle)
    assert "w" in weights.arrays
    assert not hasattr(weights, "vectors")
