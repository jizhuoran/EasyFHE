import numpy as np
from types import SimpleNamespace

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher
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


def test_constant_bundle_prepares_and_caches_vectors():
    bundle = fhe.ConstantBundle(
        vectors={"w": np.asarray([1.0, 2.0], dtype=np.double)},
        cache_mode="middle",
    )

    first = bundle.prepared_plaintext("w", slots=4, ring_dim=8)
    second = bundle.prepared_plaintext("w", slots=4, ring_dim=8)

    assert first is second
    assert first.slots == 4
    assert np.array_equal(bundle.values("w", slots=4), np.asarray([1.0, 2.0, 0.0, 0.0]))
    assert bundle.cache_info()["middle_hits"] == 1
    assert bundle.cache_info()["middle_misses"] == 1


def test_constant_bundle_accepts_prepared_vectors():
    prepared = fhe.prepare_plaintext(np.asarray([1.0, 2.0], dtype=np.double), slots=2, ring_dim=8)
    bundle = fhe.ConstantBundle(vectors={"w": prepared}, cache_mode="middle")

    assert bundle.prepared_plaintext("w", ring_dim=8) is prepared


def test_constant_bundle_preserves_complex_raw_values():
    values = np.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex128)
    bundle = fhe.ConstantBundle(vectors={"w": values}, cache_mode="none")

    padded = bundle.values("w", slots=4)

    assert padded.dtype == np.complex128
    assert np.array_equal(padded, np.asarray([1.0 + 2.0j, 3.0 - 4.0j, 0.0, 0.0]))


def test_constant_bundle_plaintext_batch_cache(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={
            "a": np.asarray([1.0], dtype=np.double),
            "b": np.asarray([2.0], dtype=np.double),
        },
        cache_mode="plain",
    )
    calls = []

    def fake_materialize(name, level, slots, crypto_context, scale, is_ext):
        calls.append((name, level, slots, scale, is_ext))
        return _cipher(name, cv_count=1)

    monkeypatch.setattr(bundle, "_materialize_plaintext", fake_materialize)

    ctx = SimpleNamespace()
    first = bundle.plaintext_batch(["a", "b"], 3, 4, ctx, is_ext=True)
    second = bundle.plaintext_batch(["a", "b"], 3, 4, ctx, is_ext=True)

    assert first is second
    assert first.batch_size == 2
    assert calls == [("a", 3, 4, 1.0, True), ("b", 3, 4, 1.0, True)]
    assert bundle.cache_info()["plain_batch_hits"] == 1
    assert bundle.cache_info()["plain_batch_misses"] == 1


def test_resnet_weight_pack_reuses_constant_bundle():
    weights = WeightPack({"w": np.asarray([1.0], dtype=np.double)}, cache_mode="middle")

    assert isinstance(weights, fhe.ConstantBundle)
    assert weights.has("w")
    assert weights.arrays is weights.vectors
