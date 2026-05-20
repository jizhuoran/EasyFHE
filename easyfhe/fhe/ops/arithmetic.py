from . import kernels as F
from ..ciphertext import Cipher
from ..runtime import validation
from . import alignment
from .primitives import _cipher_add, _cipher_add_ext, _cipher_mul, _cipher_square, _cipher_sub, _cipher_sub_ext


def _align_for_add_or_sub(in0, in1, cryptoContext):
    target = alignment.plan_add_alignment(in0, in1, cryptoContext)
    return alignment.align_to(in0, target, cryptoContext), alignment.align_to(in1, target, cryptoContext)


def _align_for_mul(ct1: Cipher, ct2: Cipher, cryptoContext):
    target1, target2 = alignment.plan_mul_alignment(ct1, ct2, cryptoContext)
    return alignment.align_to(ct1, target1, cryptoContext), alignment.align_to(ct2, target2, cryptoContext)


def homo_add(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_add", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_add_ext(in0, in1, cryptoContext)
    return _cipher_add(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_sub", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_sub_ext(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)


def homo_mul(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_mul", in0, in1, require_ext=False, require_same_metadata=("slots",))
    in0, in1 = _align_for_mul(in0, in1, cryptoContext)
    return _relinearize(_cipher_mul(in0, in1, cryptoContext), cryptoContext)


def homo_square(in0, cryptoContext):
    validation.validate_cipher_op("homo_square", in0, require_ext=False)
    in0 = alignment.align_to(in0, alignment.plan_reduce_noise_to_one(in0, cryptoContext), cryptoContext)
    return _relinearize(_cipher_square(in0, cryptoContext), cryptoContext)


def _relinearize(cipher, cryptoContext):
    key_switched = F.cv_keyswitch(
        cipher.cv[2],
        cipher.cur_limbs,
        cryptoContext.L,
        cryptoContext.mult_swk_bx,
        cryptoContext.mult_swk_ax,
        cryptoContext,
    )
    cv = [
        F.cv_add(cipher.cv[0], key_switched[0], cryptoContext.moduliQ, cipher.cur_limbs),
        F.cv_add(cipher.cv[1], key_switched[1], cryptoContext.moduliQ, cipher.cur_limbs),
    ]
    return cipher.cipher_like(cv)
