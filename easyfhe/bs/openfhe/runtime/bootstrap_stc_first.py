import math

from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import rotation
from . import approx as bootstrap_approx
from .bootstrap import (
    _mul_by_monomial_inplace,
    _raise_ciphertext,
    _scale_after_approx,
    _scale_to_original_message,
    eval_coeffs_to_slots,
    eval_slots_to_coeffs,
)


def _replicate_sparse_slots(ciphertext, slots, cryptoContext):
    for step in range(int(math.log2(cryptoContext.N // (2 * slots)))):
        ciphertext = rotation.homo_rotate_add(
            ciphertext,
            (1 << step) * slots,
            cryptoContext,
            addend=ciphertext,
        )
    return ciphertext.cipher_like(ciphertext.cv, slots=slots)


def _drop_to_stc_start(ciphertext, cryptoContext, bootstrap_plan):
    target_limbs = int(bootstrap_plan.level_budget[1]) + 2
    ciphertext = alignment.reduce_noise_to_one(ciphertext, cryptoContext)
    if ciphertext.state.cur_limbs < target_limbs:
        raise ValueError(
            "stc_first bootstrap needs at least "
            f"{target_limbs} limbs before SlotsToCoeffs, got {ciphertext.state.cur_limbs}"
        )
    if ciphertext.state.cur_limbs == target_limbs:
        return ciphertext
    return alignment.align_to(
        ciphertext,
        ciphertext.state.replace(cur_limbs=target_limbs, noise_deg=1),
        cryptoContext,
    )


def _decode_sparse_input(ciphertext, slots, cryptoContext, bootstrap_constants, bootstrap_plan):
    decoded = eval_slots_to_coeffs(ciphertext, cryptoContext, bootstrap_constants, bootstrap_plan)
    decoded = rotation.homo_rotate_add(decoded, slots, cryptoContext, addend=decoded)
    return decoded.cipher_like(decoded.cv, slots=slots)


def _eval_mod_full(encoded, cryptoContext, bootstrap_constants, bootstrap_plan):
    conjugate = rotation.homo_rotate(encoded, 2 * cryptoContext.N - 1, cryptoContext)
    imag = arithmetic.homo_sub(encoded, conjugate, cryptoContext)
    real = arithmetic.homo_add(encoded, conjugate, cryptoContext)
    imag = _mul_by_monomial_inplace(imag, 3 * cryptoContext.M // 4, cryptoContext)

    real = alignment.reduce_noise_to_one(real, cryptoContext)
    imag = alignment.reduce_noise_to_one(imag, cryptoContext)

    real = bootstrap_approx.eval_bootstrap_approx_mod(real, cryptoContext, bootstrap_constants, bootstrap_plan)
    imag = bootstrap_approx.eval_bootstrap_approx_mod(imag, cryptoContext, bootstrap_constants, bootstrap_plan)

    imag = _mul_by_monomial_inplace(imag, cryptoContext.M // 4, cryptoContext)
    encoded = arithmetic.homo_add(real, imag, cryptoContext)
    encoded = _scale_after_approx(encoded, cryptoContext, bootstrap_constants)
    return alignment.reduce_noise_to_one(encoded, cryptoContext)


def _eval_mod_sparse(encoded, cryptoContext, bootstrap_constants, bootstrap_plan):
    encoded = rotation.homo_rotate_add(encoded, 2 * cryptoContext.N - 1, cryptoContext, addend=encoded)
    encoded = alignment.reduce_noise_to_one(encoded, cryptoContext)
    encoded = bootstrap_approx.eval_bootstrap_approx_mod(encoded, cryptoContext, bootstrap_constants, bootstrap_plan)
    encoded = _scale_after_approx(encoded, cryptoContext, bootstrap_constants)
    return alignment.reduce_noise_to_one(encoded, cryptoContext)


def eval_bootstrap_stc_first(ciphertext, cryptoContext, bootstrap_constants, bootstrap_plan, L0):
    slots = bootstrap_plan.slots
    sparse = slots != cryptoContext.M // 4
    ciphertext = _drop_to_stc_start(ciphertext, cryptoContext, bootstrap_plan)

    if sparse:
        decoded = _decode_sparse_input(ciphertext, slots, cryptoContext, bootstrap_constants, bootstrap_plan)
    else:
        decoded = eval_slots_to_coeffs(ciphertext, cryptoContext, bootstrap_constants, bootstrap_plan)

    raised = _raise_ciphertext(decoded, cryptoContext, bootstrap_constants, L0)
    if sparse:
        raised = _replicate_sparse_slots(raised, slots, cryptoContext)

    encoded = eval_coeffs_to_slots(raised, cryptoContext, bootstrap_constants, bootstrap_plan)
    if sparse:
        encoded = _eval_mod_sparse(encoded, cryptoContext, bootstrap_constants, bootstrap_plan)
    else:
        encoded = _eval_mod_full(encoded, cryptoContext, bootstrap_constants, bootstrap_plan)

    encoded = _scale_to_original_message(encoded, cryptoContext, bootstrap_constants)
    return encoded.cipher_like(encoded.cv, slots=ciphertext.slots)
