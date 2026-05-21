import numpy as np

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


def test_encode_stage2_checks_middle_metadata():
    ctx = _context()
    middle = encoding.encode_stage1(np.asarray([1.0, 2.0], dtype=np.double), slots=2, ring_dim=ctx.N)

    try:
        encoding.encode_stage2(middle, level=0, slots=4, is_ext=False, cryptoContext=ctx)
    except ValueError as exc:
        assert "Prepared plaintext slots" in str(exc)
    else:
        raise AssertionError("expected middle slot mismatch to fail")
