from types import SimpleNamespace

import pytest

import easyfhe.fhe as fhe
from easyfhe.fhe.ops import fused


class FakeCipher:
    def __init__(self, name, cv_count=2):
        self.name = name
        self.cv = [object() for _ in range(cv_count)]


def test_fused_broadcast_mac_multiplies_each_item_by_shared_cipher(monkeypatch):
    calls = []

    def fake_mul_pt(cipher, plaintext, context):
        calls.append(("mul_pt", cipher.name, plaintext.name))
        return FakeCipher(f"({cipher.name}*{plaintext.name})")

    def fake_add(lhs, rhs, context):
        calls.append(("add", lhs.name, rhs.name))
        return FakeCipher(f"({lhs.name}+{rhs.name})")

    monkeypatch.setattr(fused, "homo_mul_pt", fake_mul_pt)
    monkeypatch.setattr(fused, "homo_add", fake_add)

    result = fhe.fused_broadcast_mac(
        FakeCipher("x"),
        [FakeCipher("a", cv_count=1), FakeCipher("b", cv_count=1), FakeCipher("c", cv_count=1)],
        SimpleNamespace(),
    )

    assert result.name == "(((x*a)+(x*b))+(x*c))"
    assert calls == [
        ("mul_pt", "x", "a"),
        ("mul_pt", "x", "b"),
        ("add", "(x*a)", "(x*b)"),
        ("mul_pt", "x", "c"),
        ("add", "((x*a)+(x*b))", "(x*c)"),
    ]


def test_fused_pairwise_mac_multiplies_matching_pairs(monkeypatch):
    calls = []

    def fake_mul_pt(lhs, rhs, context):
        calls.append(("mul_pt", lhs.name, rhs.name))
        return FakeCipher(f"({lhs.name}*{rhs.name})")

    def fake_add(lhs, rhs, context):
        calls.append(("add", lhs.name, rhs.name))
        return FakeCipher(f"({lhs.name}+{rhs.name})")

    monkeypatch.setattr(fused, "homo_mul_pt", fake_mul_pt)
    monkeypatch.setattr(fused, "homo_add", fake_add)

    result = fhe.fused_pairwise_mac(
        [FakeCipher("a"), FakeCipher("b")],
        [FakeCipher("x", cv_count=1), FakeCipher("y", cv_count=1)],
        SimpleNamespace(),
    )

    assert result.name == "((a*x)+(b*y))"
    assert calls == [
        ("mul_pt", "a", "x"),
        ("mul_pt", "b", "y"),
        ("add", "(a*x)", "(b*y)"),
    ]


def test_fused_pairwise_mac_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="cipher and plaintext lengths must match"):
        fhe.fused_pairwise_mac(
            [FakeCipher("a"), FakeCipher("b")],
            [FakeCipher("x")],
            SimpleNamespace(),
        )


def test_fused_mac_supports_plaintext_cipher_operands(monkeypatch):
    calls = []

    def fake_mul_pt(cipher, plaintext, context):
        calls.append(("mul_pt", cipher.name, plaintext.name))
        return FakeCipher(f"({cipher.name}*{plaintext.name})")

    monkeypatch.setattr(fused, "homo_mul_pt", fake_mul_pt)

    result = fhe.fused_pairwise_mac(
        [FakeCipher("a")],
        [FakeCipher("plain", cv_count=1)],
        SimpleNamespace(),
    )

    assert result.name == "(a*plain)"
    assert calls == [("mul_pt", "a", "plain")]
