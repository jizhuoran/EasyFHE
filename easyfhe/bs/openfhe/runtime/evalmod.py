from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import kernels as F
from easyfhe.fhe.ops import rotation

from . import approx as bootstrap_approx
from .scaling import scale_after_approx


def mul_by_monomial_inplace(cipher, monomial_degree, crypto_context):
    F.cv_mul_by_monomial(cipher.cv[0], cipher.state.cur_limbs, monomial_degree, crypto_context)
    F.cv_mul_by_monomial(cipher.cv[1], cipher.state.cur_limbs, monomial_degree, crypto_context)
    return cipher


def eval_mod_full(encoded, crypto_context, bootstrap_constants, bootstrap_plan, *, raise_to_limbs):
    conjugate = rotation.homo_rotate(encoded, 2 * crypto_context.N - 1, crypto_context)
    imag = arithmetic.homo_sub(encoded, conjugate, crypto_context)
    real = arithmetic.homo_add(encoded, conjugate, crypto_context)
    imag = mul_by_monomial_inplace(imag, 3 * crypto_context.M // 4, crypto_context)

    real = alignment.reduce_noise_to_one(real, crypto_context)
    imag = alignment.reduce_noise_to_one(imag, crypto_context)

    real = bootstrap_approx.eval_bootstrap_approx_mod(real, crypto_context, bootstrap_constants, bootstrap_plan)
    imag = bootstrap_approx.eval_bootstrap_approx_mod(imag, crypto_context, bootstrap_constants, bootstrap_plan)

    imag = mul_by_monomial_inplace(imag, crypto_context.M // 4, crypto_context)
    encoded = arithmetic.homo_add(real, imag, crypto_context)
    encoded = scale_after_approx(
        encoded,
        crypto_context,
        bootstrap_constants,
        raise_to_limbs=raise_to_limbs,
    )
    return alignment.reduce_noise_to_one(encoded, crypto_context)


def eval_mod_sparse(encoded, crypto_context, bootstrap_constants, bootstrap_plan, *, raise_to_limbs):
    encoded = rotation.homo_rotate_add(
        encoded,
        2 * crypto_context.N - 1,
        crypto_context,
        addend=encoded,
    )
    encoded = alignment.reduce_noise_to_one(encoded, crypto_context)
    encoded = bootstrap_approx.eval_bootstrap_approx_mod(
        encoded,
        crypto_context,
        bootstrap_constants,
        bootstrap_plan,
    )
    encoded = scale_after_approx(
        encoded,
        crypto_context,
        bootstrap_constants,
        raise_to_limbs=raise_to_limbs,
    )
    return alignment.reduce_noise_to_one(encoded, crypto_context)
