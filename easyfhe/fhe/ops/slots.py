import math

import easyfhe as torch

from ..ciphertext import Cipher
from .arithmetic import homo_add
from .plaintext import homo_mul_pt
from .rotation import homo_rotate


def slot_resize(x, slots, cryptoContext, *, mask=None):
    if x.is_ext:
        raise ValueError("slot_resize: expected non-ext cipher")

    if x.slots <= slots:
        res = x.deep_copy()
    else:
        if mask is None:
            raise ValueError("slot_resize requires mask plaintext when reducing slots")
        if int(mask.slots) != int(x.slots):
            raise ValueError(f"slot_resize mask slots [{mask.slots}] must match source slots [{x.slots}]")
        res = homo_mul_pt(x, mask, cryptoContext)
        for i in range(int(math.log2(slots)), int(math.log2(x.slots))):
            res = homo_add(res, homo_rotate(res, 1 << i, cryptoContext), cryptoContext)
    res.slots = slots
    return res


def extract_cv(cipher: Cipher, index, cryptoContext, append_zeros=False):
    if index not in (0, 1):
        raise ValueError(f"extract_cv: index must be 0 or 1, got {index}")
    if append_zeros:
        if index == 0:
            return cipher.cipher_like([cipher.cv[0], torch.zeros_like(cipher.cv[0])])
        return cipher.cipher_like([torch.zeros_like(cipher.cv[1]), cipher.cv[1]])
    return cipher.cipher_like([cipher.cv[index]])
