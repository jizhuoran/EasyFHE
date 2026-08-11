import math

import easyfhe as torch

from . import validation
from .arithmetic import homo_mul_pt


def expand_slots(cipher, slots, context):
    validation.validate_cipher_op("expand_slots", cipher, require_ext=False)
    slots = validation.validate_slot_count("expand_slots", slots, context)
    source_slots = validation.validate_slot_count("expand_slots", cipher.slots, context)
    if slots < source_slots:
        raise ValueError(f"expand_slots: target slots [{slots}] must be >= source slots [{source_slots}]")

    result = cipher.deep_copy()
    result.slots = slots
    return result


def fold_slots(cipher, slots, context, *, mask):
    validation.validate_cipher_op("fold_slots", cipher, require_ext=False)
    slots = validation.validate_slot_count("fold_slots", slots, context)
    source_slots = validation.validate_slot_count("fold_slots", cipher.slots, context)
    if slots >= source_slots:
        raise ValueError(f"fold_slots: target slots [{slots}] must be < source slots [{source_slots}]")
    if mask is None:
        raise ValueError("fold_slots requires an explicit mask plaintext")
    if int(mask.slots) != source_slots:
        raise ValueError(f"fold_slots mask slots [{mask.slots}] must match source slots [{source_slots}]")

    result = homo_mul_pt(cipher, mask, context)
    for level in range(_log2(slots), _log2(source_slots)):
        result = homo_rotate_add(result, 1 << level, context, addend=result)
    result.slots = slots
    return result


def pack_cipher_batch(ciphers):
    ciphers = tuple(ciphers)
    if not ciphers:
        raise ValueError("cannot pack an empty cipher batch")
    first = ciphers[0]
    for idx, cipher in enumerate(ciphers):
        if len(cipher.cv) != len(first.cv):
            raise ValueError(f"cipher batch component count mismatch at index {idx}")
        for field in ("state", "slots", "is_ext"):
            if getattr(cipher, field) != getattr(first, field):
                raise ValueError(
                    f"cipher batch {field} mismatch at index {idx}: "
                    f"{getattr(cipher, field)} != {getattr(first, field)}"
                )
    cv = [
        _pack_component([cipher.cv[component] for cipher in ciphers], first.state.cur_limbs)
        for component in range(len(first.cv))
    ]
    return first.cipher_like(cv, batch_size=len(ciphers))


def _pack_component(components, min_required_limbs):
    limb_capacity = min(int(component.shape[-2]) for component in components)
    if limb_capacity < int(min_required_limbs):
        raise ValueError(
            f"cipher batch component capacity {limb_capacity} is smaller than active limbs {min_required_limbs}"
        )
    return torch.cat([component[..., :limb_capacity, :] for component in components], dim=0)


def unpack_cipher_batch(cipher):
    if cipher.batch_size == 1 and cipher.cv[0].dim() < 3:
        return (cipher,)
    if cipher.cv[0].dim() < 3:
        raise ValueError("expected batched cipher components")
    return tuple(
        cipher.cipher_like(
            [component[index] for component in cipher.cv],
            batch_size=1,
        )
        for index in range(int(cipher.batch_size))
    )


def cipher_batch_item(cipher, index):
    if cipher.batch_size == 1 and cipher.cv[0].dim() == 2:
        return cipher
    return cipher.cipher_like([cv[int(index)] for cv in cipher.cv], batch_size=1)


def homo_rotate_add(cipher, offset, context, addend=None):
    from .rotation import homo_rotate_add as rotate_add

    return rotate_add(cipher, offset, context, addend=addend)


def _log2(slots):
    return int(math.log2(slots))
