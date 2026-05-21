import math

from ..ciphertext import Cipher, Plaintext
from . import kernels as F
from . import validation
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


def _is_encoded_scalar(value):
    return hasattr(value, "to") and hasattr(value, "dim")


def _assign_out(out, value):
    return value if out is None else out.replace_with(value)


def homo_add_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext, *, out=None):
    validation.validate_cipher_plain_op(
        "homo_add_pt",
        cipher,
        plaintext,
        require_ext=False,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_add_plain(cipher, plaintext, cryptoContext, out=out)


def homo_add_pt_inplace(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    return homo_add_pt(cipher, plaintext, cryptoContext, out=cipher)


def homo_mul_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext, *, out=None):
    validation.validate_cipher_plain_op(
        "homo_mul_pt",
        cipher,
        plaintext,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_mul_plain(cipher, plaintext, cryptoContext, out=out)


def homo_mul_pt_inplace(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    return homo_mul_pt(cipher, plaintext, cryptoContext, out=cipher)


def homo_add_scalar_double(cipher, constant, cryptoContext, *, out=None):
    validation.validate_cipher_scalar_op(
        "homo_add_scalar_double",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    if _is_encoded_scalar(constant):
        return _cipher_add_scalar(cipher, constant, cryptoContext, out=out)

    encoded_constant = _encode_double_for_scalar_op(math.fabs(constant), cipher.cur_limbs, cryptoContext)
    if constant < 0:
        result = _cipher_sub_scalar(cipher, encoded_constant, cryptoContext, out=out)
    else:
        result = _cipher_add_scalar(cipher, encoded_constant, cryptoContext, out=out)
    return result


def homo_add_scalar_double_inplace(cipher, constant, cryptoContext):
    return homo_add_scalar_double(cipher, constant, cryptoContext, out=cipher)


def homo_add_scalar_int(cipher, scalar, cryptoContext, *, out=None):
    validation.validate_cipher_scalar_op(
        "homo_add_scalar_int",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_add_scalar(cipher, scalar, cryptoContext, out=out)


def homo_add_scalar_int_inplace(cipher, scalar, cryptoContext):
    return homo_add_scalar_int(cipher, scalar, cryptoContext, out=cipher)


def homo_sub_scalar_int(cipher, scalar, cryptoContext, *, out=None):
    validation.validate_cipher_scalar_op(
        "homo_sub_scalar_int",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_sub_scalar(cipher, scalar, cryptoContext, out=out)


def homo_sub_scalar_int_inplace(cipher, scalar, cryptoContext):
    return homo_sub_scalar_int(cipher, scalar, cryptoContext, out=cipher)


def homo_mul_scalar_int(cipher, scalar, cryptoContext, *, out=None):
    validation.validate_cipher_scalar_op(
        "homo_mul_scalar_int",
        cipher,
        require_ext=False,
    )
    if _is_encoded_scalar(scalar):
        return _cipher_mul_scalar_int(cipher, scalar, cryptoContext, out=out)

    result = _cipher_mul_scalar_int(cipher, abs(scalar), cryptoContext, out=out)
    if scalar < 0:
        for component in result.cv:
            F.cv_neg(component, cryptoContext.moduliQ, result.cur_limbs, inplace=True)
    return result


def homo_mul_scalar_int_inplace(cipher, scalar, cryptoContext):
    return homo_mul_scalar_int(cipher, scalar, cryptoContext, out=cipher)


def homo_mul_scalar_double(cipher, constant, cryptoContext, *, out=None):
    validation.validate_cipher_scalar_op(
        "homo_mul_scalar_double",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    if _is_encoded_scalar(constant):
        encoded_constant = constant
    else:
        encoded_constant = _encode_double_for_scalar_op(constant, cipher.cur_limbs, cryptoContext)
    return _cipher_mul_scalar_double(cipher, encoded_constant, cryptoContext, out=out)


def homo_mul_scalar_double_inplace(cipher, constant, cryptoContext):
    return homo_mul_scalar_double(cipher, constant, cryptoContext, out=cipher)
