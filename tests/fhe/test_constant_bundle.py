import numpy as np
import pytest
from types import SimpleNamespace

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher, CipherState
from easyfhe.fhe.ops.encoding import encode_stage1, encode_stage1_packed


def _cipher(name, cv_count=2):
    cipher = Cipher(
        [torch.zeros((2, 4), dtype=torch.uint64) for _ in range(cv_count)],
        CipherState(2, 1, 1.0),
        slots=4,
        is_ext=False,
    )
    cipher.name = name
    return cipher


def _vector(values):
    return fhe.PackedRaw(torch.from_numpy(np.asarray(values)))


def test_constant_bundle_encodes_middle_and_caches_vectors(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0, 0.0, 0.0])},
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


def test_constant_bundle_accepts_cpu_tensor_vectors(monkeypatch):
    values = torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    bundle = fhe.ConstantBundle(vectors={"w": fhe.PackedRaw(values)}, cache_mode="middle")
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        calls.append(middle)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    bundle.plaintext("w", 3, 4, ctx)

    assert calls[0].slots == 4
    np.testing.assert_array_equal(calls[0].values, values.numpy())


def test_constant_bundle_accepts_unpacked_raw_with_packer(monkeypatch):
    values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    calls = []

    def packer(tensor, slots, crypto_context):
        calls.append((tensor, slots, crypto_context))
        packed = torch.zeros(slots, dtype=tensor.dtype)
        packed[: tensor.numel()] = tensor
        return packed

    bundle = fhe.ConstantBundle(vectors={"w": fhe.UnpackedRaw(values, packer)}, cache_mode="middle")
    stage2_calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        stage2_calls.append(middle)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    bundle.plaintext("w", 3, 4, ctx)

    assert calls == [(values, 4, ctx)]
    np.testing.assert_array_equal(stage2_calls[0].values, np.asarray([1.0, 2.0, 0.0, 0.0]))


def test_constant_bundle_accepts_packed_prepared_vectors(monkeypatch):
    prepared = encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=2, ring_dim=8)
    prepared.packed = True
    bundle = fhe.ConstantBundle(vectors={"w": prepared}, cache_mode="middle")
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        calls.append(middle)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    bundle.plaintext("w", 3, 2, ctx)

    assert calls == [prepared]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packed vectors require CUDA")
def test_constant_bundle_accepts_cuda_packed_tensor_vectors(monkeypatch):
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    cuda_ctx = ctx.cuda()
    slots = 8
    values = np.asarray([1.0, -2.0, 3.5], dtype=np.double)

    packed = torch.zeros(slots, dtype=torch.complex128, device="cuda")
    packed[: values.size] = torch.as_tensor(values, dtype=torch.complex128, device="cuda")

    bundle = fhe.ConstantBundle(vectors={"w": fhe.PackedRaw(packed)}, cache_mode="middle")
    direct = encode_stage1_packed(packed, cryptoContext=cuda_ctx)
    calls = []

    def fake_encode_stage2(middle, level, stage2_slots, is_ext, crypto_context):
        calls.append((middle, level, stage2_slots, is_ext, crypto_context))
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    bundle.plaintext("w", 1, slots, cuda_ctx, is_ext=True)

    middle, level, stage2_slots, is_ext, crypto_context = calls[0]
    assert level == 1
    assert stage2_slots == slots
    assert is_ext is True
    assert crypto_context is cuda_ctx
    assert middle.packed is True
    assert middle.values is packed
    np.testing.assert_array_equal(
        middle.encoded_values.cpu().numpy(),
        direct.encoded_values.cpu().numpy(),
    )
    assert middle.max_encoded_value == direct.max_encoded_value
    assert bundle.cache_info()["middle_entries"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor vectors require CUDA")
def test_constant_bundle_accepts_cuda_real_tensor_vectors(monkeypatch):
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    cuda_ctx = ctx.cuda()
    slots = 8
    values = np.asarray([1.0, -2.0, 3.5], dtype=np.double)

    tensor_values = torch.zeros(slots, dtype=torch.float64, device="cuda")
    tensor_values[: values.size] = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    complex_values = tensor_values.to(dtype=torch.complex128)

    bundle = fhe.ConstantBundle(vectors={"w": fhe.PackedRaw(tensor_values)}, cache_mode="middle")
    direct = encode_stage1_packed(complex_values, cryptoContext=cuda_ctx)
    calls = []

    def fake_encode_stage2(middle, level, stage2_slots, is_ext, crypto_context):
        calls.append((middle, level, stage2_slots, is_ext, crypto_context))
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    bundle.plaintext("w", 1, slots, cuda_ctx, is_ext=True)

    middle, level, stage2_slots, is_ext, crypto_context = calls[0]
    assert level == 1
    assert stage2_slots == slots
    assert is_ext is True
    assert crypto_context is cuda_ctx
    assert middle.packed is True
    assert middle.values.is_cuda
    assert middle.values.dtype == torch.complex128
    np.testing.assert_array_equal(
        middle.encoded_values.cpu().numpy(),
        direct.encoded_values.cpu().numpy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packed vectors require CUDA")
def test_constant_bundle_accepts_scale_alias_for_cuda_packed_tensor_vectors(monkeypatch):
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    cuda_ctx = ctx.cuda()
    packed = torch.zeros(8, dtype=torch.complex128, device="cuda")
    bundle = fhe.ConstantBundle(vectors={"w": fhe.PackedRaw(packed)}, cache_mode="none")
    calls = []

    def fake_encode_stage2(
        middle,
        level,
        slots,
        is_ext,
        crypto_context,
        *,
        scaling_factor=None,
        cur_limbs=None,
        noise_deg=1,
    ):
        calls.append((scaling_factor, cur_limbs, noise_deg))
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)

    plaintext = bundle.plaintext("w", 1, 8, cuda_ctx, scale=2.0)

    assert plaintext.name == "w"
    assert calls == [(2.0, None, 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packed vectors require CUDA")
def test_constant_bundle_mix_restores_cuda_packed_middle_after_plain_eviction(monkeypatch):
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    cuda_ctx = ctx.cuda()
    slots = 8
    large = torch.zeros(slots, dtype=torch.complex128, device="cuda")
    small = torch.zeros(slots, dtype=torch.complex128, device="cuda")
    large[0] = 1
    small[0] = 2
    bundle = fhe.ConstantBundle(
        vectors={"large": fhe.PackedRaw(large), "small": fhe.PackedRaw(small)},
        cache_mode="mix_of_middle_plain",
        plain_cache_policy="small_first",
    )
    bundle.set_plain_cache_limit_bytes(200)
    calls = []

    def fake_encode_stage2(middle, level, stage2_slots, is_ext, crypto_context):
        calls.append(middle)
        return _cipher("large", cv_count=3) if len(calls) == 1 else _cipher("small", cv_count=1)

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)

    bundle.plaintext("large", 1, slots, cuda_ctx)
    assert bundle.cache_info()["middle_entries"] == 0

    bundle.plaintext("small", 1, slots, cuda_ctx)
    info = bundle.cache_info()
    assert info["plain_cache_evictions"] == 1
    assert info["middle_entries"] == 1
    restored = next(iter(bundle._middle_cache.values()))
    assert restored.packed is True
    assert restored.values is large


def test_constant_bundle_preserves_complex_vector_values(monkeypatch):
    values = np.asarray([1.0 + 2.0j, 3.0 - 4.0j, 0.0, 0.0], dtype=np.complex128)
    bundle = fhe.ConstantBundle(vectors={"w": _vector(values)}, cache_mode="none")
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        calls.append(middle)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    bundle.plaintext("w", 3, 4, ctx)

    encoded_values = calls[0].values
    assert encoded_values.dtype == np.complex128
    assert np.array_equal(encoded_values, values)


def test_constant_bundle_rejects_unpadded_vector_values():
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0])},
        cache_mode="none",
    )

    try:
        bundle.plaintext("w", 3, 4, SimpleNamespace(N=8))
    except ValueError as exc:
        assert "must match slots" in str(exc)
        assert "pack, pad, or truncate before constructing" in str(exc)
    else:
        raise AssertionError("expected unpadded raw vector to fail")


def test_constant_bundle_accepts_scale_alias_for_vector_values(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0, 0.0, 0.0])},
        cache_mode="none",
    )
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context, *, scaling_factor=None):
        calls.append(scaling_factor)
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)

    bundle.plaintext("w", 3, 4, SimpleNamespace(N=8), scale=2.0)

    assert calls == [2.0]


def test_constant_bundle_rejects_bare_list_and_numpy_vectors():
    with pytest.raises(TypeError, match="must be PackedRaw, UnpackedRaw, or PreparedPlaintext"):
        fhe.ConstantBundle(
            vectors={"w": np.asarray([1.0, 2.0, 0.0, 0.0])},
            cache_mode="none",
        )

    with pytest.raises(TypeError, match="must be PackedRaw, UnpackedRaw, or PreparedPlaintext"):
        fhe.ConstantBundle(
            vectors={"w": [np.asarray([1.0]), np.asarray([2.0, 3.0])]},
            cache_mode="none",
        )


def test_constant_bundle_rejects_list_of_names():
    bundle = fhe.ConstantBundle(
        vectors={
            "a": _vector([1.0]),
            "b": _vector([2.0]),
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


def test_constant_bundle_named_vector_batch_matches_individual_encoding():
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    bundle = fhe.ConstantBundle(
        vectors={
            "group": _vector(
                [
                    [1.0, 2.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0],
                ],
            ),
        },
        cache_mode="none",
    )

    batch = bundle.plaintext("group", 1, 4, ctx, is_ext=True)
    first = fhe.ConstantBundle(
        vectors={"a": _vector([1.0, 2.0, 0.0, 0.0])},
        cache_mode="none",
    ).plaintext(
        "a", 1, 4, ctx, is_ext=True
    )
    second = fhe.ConstantBundle(
        vectors={"b": _vector([3.0, 4.0, 0.0, 0.0])},
        cache_mode="none",
    ).plaintext(
        "b", 1, 4, ctx, is_ext=True
    )

    assert batch.batch_size == 2
    assert batch.is_ext is True
    assert np.array_equal(batch.cv[0][0].numpy(), first.cv[0][0].numpy())
    assert np.array_equal(batch.cv[0][1].numpy(), second.cv[0][0].numpy())


def test_constant_bundle_named_vector_batch_cache(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={
            "group": _vector(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                ],
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


def test_constant_bundle_both_caches_plain_and_middle(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0, 0.0, 0.0])},
        cache_mode="both",
    )

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)
    first = bundle.plaintext("w", 3, 4, ctx)
    second = bundle.plaintext("w", 3, 4, ctx)
    info = bundle.cache_info()

    assert first is second
    assert info["plain_entries"] == 1
    assert info["middle_entries"] == 1
    assert info["plain_hits"] == 1
    assert info["middle_misses"] == 1


def test_constant_bundle_plain_cache_limit_requires_mix_of_middle_plain_mode():
    for mode in ("plain", "middle", "both"):
        try:
            fhe.ConstantBundle(vectors={"w": _vector([1.0])}, cache_mode=mode, plain_cache_limit_gb=1)
        except ValueError as exc:
            assert "cache_mode='mix_of_middle_plain'" in str(exc)
        else:
            raise AssertionError(f"expected plain cache limit to fail for {mode}")

    bundle = fhe.ConstantBundle(vectors={"w": _vector([1.0])}, cache_mode="plain")
    try:
        bundle.set_plain_cache_limit_bytes(1)
    except ValueError as exc:
        assert "cache_mode='mix_of_middle_plain'" in str(exc)
    else:
        raise AssertionError("expected plain cache limit setter to fail outside mix_of_middle_plain mode")


def test_constant_bundle_mix_of_middle_plain_restores_middle_when_plain_is_replaced(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={
            "large": _vector([1.0, 0.0, 0.0, 0.0]),
            "small": _vector([2.0, 0.0, 0.0, 0.0]),
        },
        cache_mode="mix_of_middle_plain",
        plain_cache_policy="small_first",
    )
    bundle.set_plain_cache_limit_bytes(200)

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        return _cipher("large", cv_count=3) if middle.values[0] == 1.0 else _cipher("small", cv_count=1)

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)

    bundle.plaintext("large", 3, 4, ctx)
    info = bundle.cache_info()
    assert info["plain_entries"] == 1
    assert info["middle_entries"] == 0

    bundle.plaintext("small", 3, 4, ctx)
    info = bundle.cache_info()
    assert info["plain_entries"] == 1
    assert info["middle_entries"] == 1
    assert info["plain_cache_evictions"] == 1


def test_constant_bundle_mix_of_middle_plain_keeps_middle_for_sibling_plain_variants(monkeypatch):
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0, 0.0, 0.0])},
        cache_mode="mix_of_middle_plain",
    )

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context):
        return _cipher(f"w-level-{level}")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)
    ctx = SimpleNamespace(N=8)

    bundle.plaintext("w", 3, 4, ctx)
    info = bundle.cache_info()
    assert info["plain_entries"] == 1
    assert info["middle_entries"] == 0
    assert info["middle_misses"] == 1

    bundle.plaintext("w", 2, 4, ctx)
    info = bundle.cache_info()
    assert info["plain_entries"] == 2
    assert info["middle_entries"] == 1
    assert info["middle_misses"] == 2

    bundle.plaintext("w", 1, 4, ctx)
    info = bundle.cache_info()
    assert info["plain_entries"] == 3
    assert info["middle_entries"] == 1
    assert info["middle_hits"] == 1
    assert info["middle_misses"] == 2


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


def test_constant_bundle_flexible_plaintext_requires_explicit_scaling_factor():
    ctx = SimpleNamespace(N=8, L=4, scale_mode="flexible")
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0, 0.0, 0.0])},
        cache_mode="none",
    )

    with pytest.raises(ValueError, match="requires scaling_factor"):
        bundle.plaintext("w", 3, 4, ctx)


def test_constant_bundle_plain_cache_keys_include_scaling_factor(monkeypatch):
    ctx = SimpleNamespace(N=8, L=4, scale_mode="flexible")
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0, 0.0, 0.0])},
        cache_mode="plain",
    )
    calls = []

    def fake_encode_stage2(middle, level, slots, is_ext, crypto_context, *, scaling_factor=None):
        calls.append(scaling_factor)
        return _cipher(f"w-{scaling_factor}")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)

    first = bundle.plaintext("w", 3, 4, ctx, scaling_factor=8.0)
    second = bundle.plaintext("w", 3, 4, ctx, scaling_factor=16.0)
    again = bundle.plaintext("w", 3, 4, ctx, scaling_factor=8.0)

    assert calls == [8.0, 16.0]
    assert first is again
    assert first is not second


def test_constant_bundle_plaintext_passes_explicit_cur_limbs_and_scaling_factor(monkeypatch):
    ctx = SimpleNamespace(N=8, L=4, scale_mode="flexible")
    bundle = fhe.ConstantBundle(
        vectors={"w": _vector([1.0, 2.0, 0.0, 0.0])},
        cache_mode="none",
    )
    calls = []

    def fake_encode_stage2(
        middle,
        level,
        slots,
        is_ext,
        crypto_context,
        *,
        scaling_factor=None,
        cur_limbs=None,
    ):
        calls.append((level, scaling_factor, cur_limbs))
        return _cipher("w")

    monkeypatch.setattr("easyfhe.fhe.constants.encode_stage2", fake_encode_stage2)

    bundle.plaintext("w", level=None, slots=4, cryptoContext=ctx, scaling_factor=8.0, cur_limbs=2)

    assert calls == [(None, 8.0, 2)]


def test_constant_bundle_flexible_double_scalars_require_explicit_scaling_factor():
    ctx = SimpleNamespace(
        device="cpu",
        N=8,
        L=4,
        scale_mode="flexible",
        rescale_policy="auto",
        moduliQ_scalar=[17, 19, 23, 29],
    )
    bundle = fhe.ConstantBundle(scalars={"alpha": 1.5}, cache_mode="none")

    with pytest.raises(ValueError, match="requires scaling_factor"):
        bundle.encoded_scalars("alpha", 3, 1, ctx, mode="double")

    encoded = bundle.encoded_scalars("alpha", 3, 1, ctx, mode="double", scaling_factor=8.0)

    assert encoded.tolist() == [[12, 12, 12]]
