import numpy as np

from easyfhe.fhe.ciphertext import Cipher, PreparedPlaintext
from easyfhe.fhe.ops import encoding


class _Context:
    N = 8
    L = 3


def test_encode_returns_middle_and_plaintext_for_raw_and_middle(monkeypatch):
    plaintexts = []

    def fake_make_plaintext(middle, level, slots, is_ext, context):
        plaintext = Cipher([f"pt-{len(plaintexts)}"], 3, 1.0, 1, slots, is_ext)
        plaintexts.append((middle, level, slots, is_ext, context, plaintext))
        return plaintext

    monkeypatch.setattr(encoding, "_make_plaintext", fake_make_plaintext)

    raw_middle, raw_plain = encoding.encode(
        np.asarray([1.0, 2.0], dtype=np.double),
        _Context(),
        level=2,
        slots=4,
    )
    assert isinstance(raw_middle, PreparedPlaintext)
    assert raw_plain is plaintexts[0][-1]

    middle, plain = encoding.encode(raw_middle, _Context(), level=3, slots=4, is_ext=True)
    assert middle is raw_middle
    assert plain is plaintexts[1][-1]
    assert plaintexts[1][1:4] == (3, 4, True)


def test_encode_passthrough_plaintext_returns_no_middle():
    plain = Cipher(["cv0"], 3, 1.0, 1, 4, False)

    middle, result = encoding.encode(plain, _Context(), level=0, slots=4)

    assert middle is None
    assert result is plain


def test_encode_checks_existing_metadata():
    middle = PreparedPlaintext(
        np.asarray([1.0, 2.0], dtype=np.double),
        2,
        np.asarray([1.0, 0.0, 2.0, 0.0], dtype=np.double),
        2.0,
    )
    plain = Cipher(["cv0"], 3, 1.0, 1, 4, False)

    try:
        encoding.encode(middle, _Context(), level=0, slots=4)
    except ValueError as exc:
        assert "Prepared plaintext slots" in str(exc)
    else:
        raise AssertionError("expected middle slot mismatch to fail")

    try:
        encoding.encode(plain, _Context(), level=0, slots=8)
    except ValueError as exc:
        assert "Plaintext slots" in str(exc)
    else:
        raise AssertionError("expected plaintext slot mismatch to fail")

    try:
        encoding.encode(plain, _Context(), level=1, slots=4)
    except ValueError as exc:
        assert "cur_limbs" in str(exc)
    else:
        raise AssertionError("expected plaintext level mismatch to fail")


def test_encode_rejects_legacy_positional_signature():
    try:
        encoding.encode(np.asarray([1.0], dtype=np.double), "name", 0, 1, False, _Context())
    except TypeError:
        pass
    else:
        raise AssertionError("expected legacy encode signature to fail")
