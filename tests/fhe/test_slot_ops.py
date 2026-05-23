from types import SimpleNamespace

import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import CipherState
from easyfhe.fhe.ops import layout as layout_ops


class _DummyCipher:
    is_ext = False

    def __init__(self, slots):
        self.slots = slots
        self.state = CipherState(3, 1, None)

    def deep_copy(self):
        copied = _DummyCipher(self.slots)
        copied.state = self.state
        copied.is_ext = self.is_ext
        return copied


def test_expand_slots_rejects_reducing_slots():
    ctx = SimpleNamespace(L=5, N=32)

    try:
        fhe.expand_slots(_DummyCipher(16), 4, ctx)
    except ValueError as exc:
        assert "must be >= source slots" in str(exc)
    else:
        raise AssertionError("expected expand_slots to reject reducing slots")


def test_fold_slots_requires_mask_when_reducing_slots():
    ctx = SimpleNamespace(L=5, N=32)

    try:
        fhe.fold_slots(_DummyCipher(16), 4, ctx, mask=None)
    except ValueError as exc:
        assert "requires an explicit mask plaintext" in str(exc)
    else:
        raise AssertionError("expected fold_slots without mask to fail")


def test_fold_slots_uses_explicit_mask_when_reducing_slots(monkeypatch):
    monkeypatch.setattr(layout_ops, "homo_mul_pt", lambda x, mask, ctx: x.deep_copy())
    monkeypatch.setattr(layout_ops, "homo_rotate_add", lambda x, offset, ctx, addend=None: x.deep_copy())

    ctx = SimpleNamespace(L=5, N=32)
    mask = SimpleNamespace(slots=16, is_ext=False)
    resized = fhe.fold_slots(_DummyCipher(16), 4, ctx, mask=mask)

    assert resized.slots == 4


def test_fold_slots_checks_explicit_mask_slots():
    ctx = SimpleNamespace(L=5, N=32)
    mask = SimpleNamespace(slots=4, is_ext=False)

    try:
        fhe.fold_slots(_DummyCipher(16), 4, ctx, mask=mask)
    except ValueError as exc:
        assert "must match source slots" in str(exc)
    else:
        raise AssertionError("expected fold_slots mask slot mismatch to fail")
