import math

import easyfhe as torch

from ..ciphertext import Cipher
from ..runtime.instrumentation import run_instrumented_op
from .arithmetic import homo_add
from .encoding import encode
from .plaintext import homo_mul_pt
from .rotation import homo_rotate


def slot_resize(x, slots, cryptoContext):
    return run_instrumented_op(cryptoContext, "slot_resize", _slot_resize, x, slots, cryptoContext)


def _slot_resize(x, slots, cryptoContext):
    if x.is_ext:
        raise ValueError("slot_resize: expected non-ext cipher")

    if x.slots <= slots:
        res = x.deep_copy()
    else:
        mask_name = "slot_conversion_mask_{}to{}".format(x.slots, slots)
        _, mask = encode(
            cryptoContext.encode_values[mask_name],
            cryptoContext,
            level=cryptoContext.L - x.cur_limbs,
            slots=x.slots,
            is_ext=x.is_ext,
        )
        res = homo_mul_pt(x, mask, cryptoContext)
        for i in range(int(math.log2(slots)), int(math.log2(x.slots))):
            res = homo_add(res, homo_rotate(res, 1 << i, cryptoContext), cryptoContext)
    res.slots = slots
    return res


def extract_cv(cipher: Cipher, index, cryptoContext, append_zeros=False):
    return run_instrumented_op(cryptoContext, "extract_cv", _extract_cv, cipher, index, cryptoContext, append_zeros=append_zeros)


def _extract_cv(cipher: Cipher, index, cryptoContext, append_zeros=False):
    if index not in (0, 1):
        raise ValueError(f"extract_cv: index must be 0 or 1, got {index}")
    if append_zeros:
        if index == 0:
            return cipher.cipher_like([cipher.cv[0], torch.zeros_like(cipher.cv[0])])
        return cipher.cipher_like([torch.zeros_like(cipher.cv[1]), cipher.cv[1]])
    return cipher.cipher_like([cipher.cv[index]])
