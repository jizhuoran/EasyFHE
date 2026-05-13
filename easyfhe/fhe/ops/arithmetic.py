from . import kernels as F
from ..ciphertext import Cipher
from ..runtime import validation
from ..runtime.instrumentation import run_instrumented_op
from . import alignment
from .primitives import _cipher_add, _cipher_add_ext, _cipher_mul, _cipher_square, _cipher_sub, _cipher_sub_ext


def _align_for_add_or_sub(in0, in1, cryptoContext):
    target = alignment.plan_add_alignment(in0, in1, cryptoContext)
    return alignment.align_to(in0, target, cryptoContext), alignment.align_to(in1, target, cryptoContext)


def _align_for_mul(ct1: Cipher, ct2: Cipher, cryptoContext):
    target1, target2 = alignment.plan_mul_alignment(ct1, ct2, cryptoContext)
    return alignment.align_to(ct1, target1, cryptoContext), alignment.align_to(ct2, target2, cryptoContext)


def homo_add(in0, in1, cryptoContext):
    return run_instrumented_op(cryptoContext, "homo_add", _homo_add, in0, in1, cryptoContext)


def _homo_add(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_add", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_add_ext(in0, in1, cryptoContext)
    return _cipher_add(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    return run_instrumented_op(cryptoContext, "homo_sub", _homo_sub, in0, in1, cryptoContext)


def _homo_sub(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_sub", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_sub_ext(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)


def homo_mul(in0, in1, cryptoContext):
    return run_instrumented_op(cryptoContext, "homo_mul", _homo_mul, in0, in1, cryptoContext)


def _homo_mul(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_mul", in0, in1, require_ext=False, require_same_metadata=("slots",))
    in0, in1 = _align_for_mul(in0, in1, cryptoContext)
    res = _cipher_mul(in0, in1, cryptoContext)
    tmp = res.cipher_like(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.L,
            cryptoContext.mult_swk_bx,
            cryptoContext.mult_swk_ax,
            cryptoContext,
        )
    )
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)


def homo_square(in0, cryptoContext):
    return run_instrumented_op(cryptoContext, "homo_square", _homo_square, in0, cryptoContext)


def _homo_square(in0, cryptoContext):
    validation.validate_cipher_op("homo_square", in0, require_ext=False)
    in0 = alignment.align_to(in0, alignment.plan_reduce_noise_to_one(in0, cryptoContext), cryptoContext)
    res = _cipher_square(in0, cryptoContext)
    tmp = res.cipher_like(
        F.cv_keyswitch(
            res.cv[2],
            res.cur_limbs,
            cryptoContext.L,
            cryptoContext.mult_swk_bx,
            cryptoContext.mult_swk_ax,
            cryptoContext,
        )
    )
    res.cv = res.cv[:2]
    return _cipher_add(res, tmp, cryptoContext)
