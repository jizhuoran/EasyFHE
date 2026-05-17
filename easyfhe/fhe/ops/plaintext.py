import math

from ..ciphertext import Cipher, Plaintext
from ..runtime import validation
from .primitives import (
    _cipher_add_plain,
    _cipher_add_scalar,
    _cipher_mul_plain,
    _cipher_mul_scalar_double,
    _cipher_mul_scalar_int,
    _cipher_neg,
    _cipher_sub_scalar,
)


def _reduce_scalar_to_crt(value, cur_limbs, moduli):
    return [int(value) % int(moduli[i]) for i in range(cur_limbs)]


def _encode_double_for_scalar_op(constant, cur_limbs, cryptoContext):
    scale = cryptoContext.scale_at(cur_limbs)
    encoded = int(constant * scale + 0.5)
    return _reduce_scalar_to_crt(encoded, cur_limbs, cryptoContext.moduliQ_scalar)


def homo_add_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    return _homo_add_pt(cipher, plaintext, cryptoContext)


def _homo_add_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    validation.validate_cipher_plain_op(
        "homo_add_pt",
        cipher,
        plaintext,
        require_ext=False,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_add_plain(cipher, plaintext, cryptoContext)


def homo_mul_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    return _homo_mul_pt(cipher, plaintext, cryptoContext)


def _homo_mul_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    validation.validate_cipher_plain_op(
        "homo_mul_pt",
        cipher,
        plaintext,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_mul_plain(cipher, plaintext, cryptoContext)


def homo_add_scalar_double(cipher, constant, cryptoContext):
    return _homo_add_scalar_double(cipher, constant, cryptoContext)


def _homo_add_scalar_double(cipher, constant, cryptoContext):
    validation.validate_cipher_scalar_op(
        "homo_add_scalar_double",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    encoded_constant = _encode_double_for_scalar_op(
        math.fabs(constant),
        cipher.cur_limbs,
        cryptoContext,
    )
    if constant < 0:
        result = _cipher_sub_scalar(cipher, encoded_constant, cryptoContext)
    else:
        result = _cipher_add_scalar(cipher, encoded_constant, cryptoContext)
    return result


def homo_add_scalar_int(cipher, scalar, cryptoContext):
    return _homo_add_scalar_int(cipher, scalar, cryptoContext)


def _homo_add_scalar_int(cipher, scalar, cryptoContext):
    validation.validate_cipher_scalar_op(
        "homo_add_scalar_int",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_add_scalar(cipher, scalar, cryptoContext)


def homo_mul_scalar_int(cipher, scalar, cryptoContext):
    return _homo_mul_scalar_int(cipher, scalar, cryptoContext)


def _homo_mul_scalar_int(cipher, scalar, cryptoContext):
    validation.validate_cipher_scalar_op(
        "homo_mul_scalar_int",
        cipher,
        require_ext=False,
    )
    result = _cipher_mul_scalar_int(cipher, abs(scalar), cryptoContext)
    if scalar < 0:
        result = _cipher_neg(result, cryptoContext)
    return result


def homo_mul_scalar_double(cipher, constant, cryptoContext):
    return _homo_mul_scalar_double(cipher, constant, cryptoContext)


def _homo_mul_scalar_double(cipher, constant, cryptoContext):
    validation.validate_cipher_scalar_op(
        "homo_mul_scalar_double",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    encoded_constant = _encode_double_for_scalar_op(constant, cipher.cur_limbs, cryptoContext)
    result = _cipher_mul_scalar_double(cipher, encoded_constant, cryptoContext)
    return result
