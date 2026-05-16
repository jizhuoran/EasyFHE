import numpy as np

import easyfhe.fhe as fhe
from examples.resnet20_aespa.weight_pack import WeightPack


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


def test_resnet_weight_pack_reuses_constant_bundle():
    weights = WeightPack({"w": np.asarray([1.0], dtype=np.double)}, cache_mode="middle")

    assert isinstance(weights, fhe.ConstantBundle)
    assert weights.has("w")
    assert weights.arrays is weights.vectors
