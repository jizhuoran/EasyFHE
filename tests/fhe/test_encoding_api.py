import numpy as np
import pytest

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ops import encoding


def _context():
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    return ctx


def test_encode_stage1_accepts_single_raw_vector():
    middle = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=4, ring_dim=8)

    assert isinstance(middle, encoding.PreparedPlaintext)
    assert middle.slots == 4
    assert middle.values.shape == (4,)
    assert middle.encoded_values.shape == (8,)


def test_encode_stage1_accepts_raw_batch():
    middle = encoding.encode_stage1(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.double),
        slots=4,
        ring_dim=8,
    )

    assert isinstance(middle, encoding.PreparedPlaintext)
    assert middle.slots == 4
    assert middle.values.shape == (2, 4)
    assert middle.encoded_values.shape == (2, 8)


def test_encode_stage1_uses_context_encode_tables():
    ctx = _context()
    values = np.asarray([1.0, -2.0, 3.5], dtype=np.double)
    from_ring_dim = encoding.encode_stage1(values, slots=8, ring_dim=ctx.N)
    from_context = encoding.encode_stage1(values, slots=8, cryptoContext=ctx)

    np.testing.assert_allclose(
        from_context.encoded_values.numpy(),
        from_ring_dim.encoded_values.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert from_context.max_encoded_value == pytest.approx(from_ring_dim.max_encoded_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stage1 encode requires CUDA")
def test_encode_stage1_cuda_matches_cpu_for_single_and_batch():
    ctx = _context()
    cuda_ctx = ctx.cuda()
    single_values = np.asarray([1.0, -2.0, 3.5], dtype=np.double)
    cpu_single = encoding.encode_stage1(single_values, slots=8, cryptoContext=ctx)
    cuda_single = encoding.encode_stage1(single_values, slots=8, cryptoContext=cuda_ctx)

    np.testing.assert_allclose(
        cuda_single.encoded_values.cpu().numpy(),
        cpu_single.encoded_values.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert cuda_single.values.shape == cpu_single.values.shape
    assert cuda_single.slots == cpu_single.slots
    assert cuda_single.max_encoded_value == pytest.approx(cpu_single.max_encoded_value)

    batch_values = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [-4.0, 0.5, 6.0],
        ],
        dtype=np.double,
    )
    cpu_batch = encoding.encode_stage1(batch_values, slots=8, cryptoContext=ctx)
    cuda_batch = encoding.encode_stage1(batch_values, slots=8, cryptoContext=cuda_ctx)

    np.testing.assert_allclose(
        cuda_batch.encoded_values.cpu().numpy(),
        cpu_batch.encoded_values.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert cuda_batch.values.shape == cpu_batch.values.shape
    assert cuda_batch.slots == cpu_batch.slots
    assert cuda_batch.max_encoded_value == pytest.approx(cpu_batch.max_encoded_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packed stage1 encode requires CUDA")
def test_encode_stage1_packed_matches_raw_cuda_path():
    ctx = _context()
    cuda_ctx = ctx.cuda()

    values = np.asarray([1.0, -2.0, 3.5], dtype=np.double)
    raw = encoding.encode_stage1(values, slots=8, cryptoContext=cuda_ctx)

    packed = torch.zeros(8, dtype=torch.complex128, device="cuda")
    packed[: values.size] = torch.as_tensor(values, dtype=torch.complex128, device="cuda")
    prepared = encoding.encode_stage1_packed(packed, cryptoContext=cuda_ctx)

    assert prepared.packed is True
    assert prepared.values is packed
    assert prepared.slots == 8
    assert prepared.encoded_values.shape == raw.encoded_values.shape
    np.testing.assert_allclose(
        prepared.encoded_values.cpu().numpy(),
        raw.encoded_values.cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert prepared.max_encoded_value == pytest.approx(raw.max_encoded_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packed stage1 encode requires CUDA")
def test_encode_stage1_packed_accepts_batch_and_complex_dtypes():
    ctx = _context()
    cuda_ctx = ctx.cuda()
    values = np.asarray([[1.0, 2.0, 0.5], [-4.0, 0.5, 6.0]], dtype=np.double)
    raw = encoding.encode_stage1(values, slots=8, cryptoContext=cuda_ctx)

    dtypes = [torch.complex64, torch.complex128]
    if hasattr(torch, "complex32"):
        dtypes.insert(0, torch.complex32)
    for dtype in dtypes:
        packed = torch.zeros((2, 8), dtype=dtype, device="cuda")
        packed[:, : values.shape[1]] = torch.as_tensor(values, dtype=dtype, device="cuda")
        prepared = encoding.encode_stage1_packed(packed, slots=8, cryptoContext=cuda_ctx)

        assert prepared.packed is True
        assert prepared.values is packed
        assert prepared.encoded_values.shape == raw.encoded_values.shape
        np.testing.assert_allclose(
            prepared.encoded_values.cpu().numpy(),
            raw.encoded_values.cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused batch encode requires CUDA")
def test_fused_encode_batch_matches_stage1_stage2_cuda():
    _, cuda_ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=16, dnum=1, dcrt_bits=50, first_mod=55),
        device="cuda",
    )
    rng = np.random.default_rng(0)
    values = (
        rng.standard_normal((2, 32768)) + 1j * rng.standard_normal((2, 32768))
    ).astype(np.complex128) * 1e-3
    packed = torch.from_numpy(values).to("cuda")
    level = cuda_ctx.L - 2

    old = encoding.encode_stage2(
        encoding.encode_stage1_packed(packed, cryptoContext=cuda_ctx),
        level=level,
        slots=32768,
        is_ext=False,
        cryptoContext=cuda_ctx,
    )
    fast = encoding.fused_encode_batch(
        packed,
        level=level,
        slots=32768,
        cryptoContext=cuda_ctx,
    )

    assert fast.batch_size == 2
    assert fast.state == old.state
    np.testing.assert_array_equal(
        fast.cv[0].cpu().numpy(),
        old.cv[0].cpu().numpy(),
    )


def test_encode_stage2_materializes_single_and_batch_plaintexts():
    ctx = _context()
    single = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=4, ring_dim=ctx.N)
    batch = encoding.encode_stage1(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.double),
        slots=4,
        ring_dim=ctx.N,
    )

    single_pt = encoding.encode_stage2(single, level=1, slots=4, is_ext=False, cryptoContext=ctx)
    batch_pt = encoding.encode_stage2(batch, level=1, slots=4, is_ext=True, cryptoContext=ctx)

    assert single_pt.batch_size == 1
    assert single_pt.is_ext is False
    assert batch_pt.batch_size == 2
    assert batch_pt.is_ext is True


def test_flexible_encode_stage2_requires_explicit_scaling_factor():
    middle = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=4, ring_dim=8)
    ctx = type("Ctx", (), {"L": 4, "scale_mode": "flexible"})()

    with pytest.raises(ValueError, match="requires scaling_factor"):
        encoding.encode_stage2(middle, level=1, slots=4, is_ext=False, cryptoContext=ctx)


def test_flexible_encode_stage2_uses_explicit_scaling_factor(monkeypatch):
    middle = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=4, ring_dim=8)
    ctx = type("Ctx", (), {"L": 4, "N": 8, "scale_mode": "flexible"})()
    calls = {}

    def fake_cv_encode(encoded_values, ring_dim, cur_limbs, slots, scaling_factor, is_ext, context):
        calls["args"] = (ring_dim, cur_limbs, slots, scaling_factor, is_ext, context)
        return torch.zeros((1, cur_limbs, ring_dim), dtype=torch.uint64)

    monkeypatch.setattr(encoding.F, "cv_encode", fake_cv_encode)

    plaintext = encoding.encode_stage2(
        middle,
        level=1,
        slots=4,
        is_ext=False,
        cryptoContext=ctx,
        scaling_factor=13.0,
    )

    assert calls["args"] == (8, 3, 4, 13.0, False, ctx)
    assert plaintext.state == fhe.CipherState(3, 1, 13.0)


def test_encode_stage2_accepts_explicit_cur_limbs(monkeypatch):
    middle = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=4, ring_dim=8)
    ctx = type("Ctx", (), {"L": 4, "N": 8, "scale_mode": "flexible"})()
    calls = {}

    def fake_cv_encode(encoded_values, ring_dim, cur_limbs, slots, scaling_factor, is_ext, context):
        calls["cur_limbs"] = cur_limbs
        return torch.zeros((1, cur_limbs, ring_dim), dtype=torch.uint64)

    monkeypatch.setattr(encoding.F, "cv_encode", fake_cv_encode)

    plaintext = encoding.encode_stage2(
        middle,
        level=None,
        slots=4,
        is_ext=False,
        cryptoContext=ctx,
        cur_limbs=2,
        scaling_factor=17.0,
    )

    assert calls["cur_limbs"] == 2
    assert plaintext.state == fhe.CipherState(2, 1, 17.0)


def test_encode_stage2_rejects_inconsistent_level_and_cur_limbs():
    middle = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=4, ring_dim=8)
    ctx = type("Ctx", (), {"L": 4, "scale_mode": "flexible"})()

    with pytest.raises(ValueError, match="inconsistent level and cur_limbs"):
        encoding.encode_stage2(
            middle,
            level=1,
            slots=4,
            is_ext=False,
            cryptoContext=ctx,
            cur_limbs=2,
            scaling_factor=17.0,
        )


def test_client_encrypt_decrypt_roundtrip_cpu():
    client, _ = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=5, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    values = np.asarray([0.125, -0.25, 0.5, 1.0], dtype=np.double)
    cipher = client.encrypt(values, device="cpu", level=0, slots=8)
    decoded = client.decrypt(cipher).cpu().numpy()

    np.testing.assert_allclose(decoded[: values.size], values, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decrypt/decode requires CUDA")
def test_client_encrypt_decrypt_roundtrip_cuda():
    client, _ = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=14, dnum=1, dcrt_bits=30, first_mod=35),
        device="cuda",
    )
    values = np.asarray([0.125, -0.25, 0.5, 1.0], dtype=np.double)
    cipher = client.encrypt(values, device="cuda", level=0, slots=8)
    decoded = client.decrypt(cipher).cpu().numpy()

    np.testing.assert_allclose(decoded[: values.size], values, rtol=1e-4, atol=1e-4)


def test_encode_stage2_checks_middle_metadata():
    ctx = _context()
    middle = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=2, ring_dim=ctx.N)

    try:
        encoding.encode_stage2(middle, level=0, slots=4, is_ext=False, cryptoContext=ctx)
    except ValueError as exc:
        assert "Prepared plaintext slots" in str(exc)
    else:
        raise AssertionError("expected middle slot mismatch to fail")
