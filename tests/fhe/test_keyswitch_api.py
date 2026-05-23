from types import SimpleNamespace

import pytest

from easyfhe.fhe.ciphertext import Cipher, CipherState
from easyfhe.fhe.ops import rotation


def test_modup_to_ext_uses_second_cipher_component(monkeypatch):
    seen = {}

    def fake_cv_modup(cv, cur_limbs, context):
        seen["cv"] = cv
        seen["cur_limbs"] = cur_limbs
        seen["context"] = context
        return ("modup", cv)

    monkeypatch.setattr(rotation.F, "cv_modup", fake_cv_modup)

    context = SimpleNamespace()
    cipher = Cipher(["cv0", "cv1"], CipherState(3, 1, 1.0), slots=8, is_ext=False)

    result = rotation._modup_to_ext(cipher, context)

    assert seen == {"cv": "cv1", "cur_limbs": 3, "context": context}
    assert result.cv == [("modup", "cv1")]
    assert result.is_ext is True


def test_modup_to_ext_rejects_single_component_cipher():
    context = SimpleNamespace()
    cipher = Cipher(["cv0"], CipherState(3, 1, 1.0), slots=8, is_ext=False)

    with pytest.raises(ValueError, match="expected at least two components"):
        rotation._modup_to_ext(cipher, context)
