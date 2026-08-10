import math

from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import rotation
from .evalmod import eval_mod_full, eval_mod_sparse
from .raising import raise_ciphertext
from .scaling import scale_to_original_message
from .transforms import eval_coeffs_to_slots, eval_slots_to_coeffs


def _replicate_sparse_slots(raised, slots, cryptoContext):
    for step in range(int(math.log2(cryptoContext.N // (2 * slots)))):
        raised = rotation.homo_rotate_add(
            raised,
            (1 << step) * slots,
            cryptoContext,
            addend=raised,
        )
    return raised.cipher_like(raised.cv, slots=slots)


def _restore_sparse_slots(decoded, slots, original_slots, cryptoContext):
    decoded = rotation.homo_rotate_add(decoded, slots, cryptoContext, addend=decoded)
    decoded = decoded.cipher_like(decoded.cv, slots=slots)
    return decoded.cipher_like(decoded.cv, slots=original_slots)


def _bootstrap_fully_packed(raised, cryptoContext, bootstrap_constants, bootstrap_plan, *, active_l0=None):
    encoded = eval_coeffs_to_slots(raised, cryptoContext, bootstrap_constants, bootstrap_plan)
    encoded = eval_mod_full(
        encoded,
        cryptoContext,
        bootstrap_constants,
        bootstrap_plan,
        active_l0=active_l0,
    )
    return eval_slots_to_coeffs(encoded, cryptoContext, bootstrap_constants, bootstrap_plan)


def _bootstrap_sparse(
    raised,
    original_slots,
    slots,
    cryptoContext,
    bootstrap_constants,
    bootstrap_plan,
    *,
    active_l0=None,
):
    raised = _replicate_sparse_slots(raised, slots, cryptoContext)

    encoded = eval_coeffs_to_slots(raised, cryptoContext, bootstrap_constants, bootstrap_plan)
    encoded = eval_mod_sparse(
        encoded,
        cryptoContext,
        bootstrap_constants,
        bootstrap_plan,
        active_l0=active_l0,
    )

    decoded = eval_slots_to_coeffs(encoded, cryptoContext, bootstrap_constants, bootstrap_plan)
    return _restore_sparse_slots(decoded, slots, original_slots, cryptoContext)


def eval_bootstrap_modraise_first(
    ciphertext,
    cryptoContext,
    bootstrap_constants,
    bootstrap_plan,
    L0,
    *,
    active_l0=None,
):
    active_l0 = int(L0 if active_l0 is None else active_l0)
    slots = bootstrap_plan.slots
    raised = raise_ciphertext(
        ciphertext,
        cryptoContext,
        bootstrap_constants,
        L0,
        active_l0=active_l0,
    )
    if slots == cryptoContext.M // 4:
        decoded = _bootstrap_fully_packed(
            raised,
            cryptoContext,
            bootstrap_constants,
            bootstrap_plan,
            active_l0=active_l0,
        )
    else:
        decoded = _bootstrap_sparse(
            raised,
            ciphertext.slots,
            slots,
            cryptoContext,
            bootstrap_constants,
            bootstrap_plan,
            active_l0=active_l0,
        )

    decoded = scale_to_original_message(decoded, cryptoContext, bootstrap_constants)
    return decoded.cipher_like(decoded.cv, slots=ciphertext.slots)
