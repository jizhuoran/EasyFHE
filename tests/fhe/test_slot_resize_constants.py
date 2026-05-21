from types import SimpleNamespace

import easyfhe.fhe as fhe
from easyfhe.fhe.ops import slots as slots_ops


class _DummyCipher:
    is_ext = False
    cur_limbs = 3
    noise_deg = 1

    def __init__(self, slots):
        self.slots = slots

    def deep_copy(self):
        copied = _DummyCipher(self.slots)
        copied.cur_limbs = self.cur_limbs
        copied.noise_deg = self.noise_deg
        copied.is_ext = self.is_ext
        return copied


def test_slot_resize_requires_mask_when_reducing_slots():
    ctx = SimpleNamespace(L=5, N=32)

    try:
        fhe.slot_resize(_DummyCipher(16), 4, ctx)
    except ValueError as exc:
        assert "requires mask plaintext" in str(exc)
    else:
        raise AssertionError("expected slot_resize without mask to fail when reducing slots")


def test_slot_resize_uses_explicit_mask_when_reducing_slots(monkeypatch):
    monkeypatch.setattr(slots_ops, "homo_mul_pt", lambda x, mask, ctx: x.deep_copy())
    monkeypatch.setattr(slots_ops, "homo_rotate", lambda x, offset, ctx: x.deep_copy())
    monkeypatch.setattr(slots_ops, "homo_add", lambda a, b, ctx: a.deep_copy())

    ctx = SimpleNamespace(L=5, N=32)
    mask = SimpleNamespace(slots=16, is_ext=False)
    resized = fhe.slot_resize(_DummyCipher(16), 4, ctx, mask=mask)

    assert resized.slots == 4


def test_slot_resize_checks_explicit_mask_slots():
    ctx = SimpleNamespace(L=5, N=32)
    mask = SimpleNamespace(slots=4, is_ext=False)

    try:
        fhe.slot_resize(_DummyCipher(16), 4, ctx, mask=mask)
    except ValueError as exc:
        assert "must match source slots" in str(exc)
    else:
        raise AssertionError("expected slot_resize mask slot mismatch to fail")
