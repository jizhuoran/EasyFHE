import math
from dataclasses import dataclass

import numpy as np

from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import homo
from easyfhe.fhe.ops import kernels as F
from easyfhe.fhe.ops import rotation
from easyfhe.fhe.ops.primitives import _cipher_add_scalar, _cipher_sub_scalar

from .approx_plan import ChebyshevPSNode, degree, get_bootstrap_approx_plan


@dataclass(frozen=True)
class _ChebyshevBasis:
    items: tuple
    batch: object


def _add_named_scalar_double_preserve_noise(ciphertext, name, constants, cryptoContext):
    constant = constants.scalar(name)
    encoded = constants.encoded_scalars(
        name,
        ciphertext.cur_limbs,
        ciphertext.noise_deg,
        cryptoContext,
        absolute=True,
    )[0]
    if constant < 0:
        return _cipher_sub_scalar(ciphertext, encoded, cryptoContext)
    return _cipher_add_scalar(ciphertext, encoded, cryptoContext)


def _batch_prefix(cipher, size):
    size = int(size)
    if size == cipher.batch_size:
        return cipher
    return cipher.cipher_like(
        [component[:size] for component in cipher.cv],
        batch_size=size,
        cipher_id="assign",
    )


def _scalar_weighted_acc(cipher_batch, scalars, cryptoContext):
    cv = F.cipher_scalar_weighted_acc(cipher_batch, scalars, cryptoContext)
    return cipher_batch.cipher_like(
        list(cv),
        scaling_factor=cipher_batch.scaling_factor * cryptoContext.scale_at(cipher_batch.cur_limbs),
        noise_deg=cipher_batch.noise_deg + 1,
        batch_size=1,
        cipher_id="assign",
    )


def _truncated_degree(coefficients, size):
    truncated = np.copy(coefficients[:size])
    truncated.resize(size, refcheck=False)
    return degree(truncated)


def _chebyshev_tail(
    coefficients,
    T,
    cryptoContext,
    constants,
    bootstrap_plan,
    scalar_path,
    *,
    size=None,
    identity_shortcut=False,
):
    deg = degree(coefficients) if size is None else _truncated_degree(coefficients, size)
    if deg < 1:
        return None
    if identity_shortcut and deg == 1 and coefficients[1] == 1:
        return T.items[0]

    names = bootstrap_plan.approx_scalar_names[tuple(scalar_path)]
    if len(names) != int(deg):
        raise ValueError(
            f"Chebyshev scalar metadata mismatch at {scalar_path}: "
            f"got {len(names)}, expected {deg}"
        )

    batch = _batch_prefix(T.batch, deg)
    scalars = constants.encoded_scalars(names, batch.cur_limbs, 1, cryptoContext, mode="double")
    return alignment.rescale_one_level(
        _scalar_weighted_acc(batch, scalars, cryptoContext),
        cryptoContext,
    )


def _add_chebyshev_constant(value, coefficients, cryptoContext):
    return homo.homo_add_scalar_double(value, coefficients[0] / 2, cryptoContext)


def _chebyshev_basis(x, a, b, k, cryptoContext):
    T = [_scale_input_to_unit_interval(x, a, b, cryptoContext)]
    for order in range(2, k + 1):
        prod = homo.homo_mul(T[order // 2 - 1], T[(order + 1) // 2 - 1], cryptoContext)
        value = homo.homo_add(prod, prod, cryptoContext)
        value = alignment.rescale_one_level(value, cryptoContext)
        if order & 1:
            value = homo.homo_sub(value, T[0], cryptoContext)
        else:
            value = homo.homo_add_scalar_double(value, -1.0, cryptoContext)
        T.append(value)

    final = T[-1]
    # min_limbs = min(item.cur_limbs for item in T)
    # if final.cur_limbs != min_limbs:
    #     raise ValueError(
    #         f"Chebyshev basis target is not the lowest limb state: "
    #         f"final={final.cur_limbs}, min={min_limbs}"
    #     )
    # if any(item.noise_deg != 1 for item in T):
    #     raise ValueError(
    #         "Chebyshev basis expects all terms to have noise_deg=1 before packing: "
    #         f"{[item.noise_deg for item in T]}"
    #     )
    items = tuple(
        alignment.align_to(
            item,
            alignment.CipherState(final.cur_limbs, final.noise_deg, final.scaling_factor),
            cryptoContext,
        )
        for item in T
    )
    return _ChebyshevBasis(items=items, batch=rotation._pack_ciphers(items))


def _scale_input_to_unit_interval(x, a, b, cryptoContext):
    y = x
    alpha = 2 / (b - a)
    if not math.isclose(alpha, 1.0):
        y = homo.homo_mul_scalar_double(x, alpha, cryptoContext)
        y = alignment.rescale_one_level(y, cryptoContext)

    beta = 2 * a / (b - a)
    if not math.isclose(beta, -1.0):
        y = homo.homo_add_scalar_double(y, -1.0 - beta, cryptoContext)
    return y


def _chebyshev_doubling_basis(Tk, m, cryptoContext):
    # T2[i] = T_{k * 2^i}(x). T2[0] is T_k(x).
    T2 = [Tk]
    for _ in range(1, m):
        value = homo.homo_square(T2[-1], cryptoContext)
        value = homo.homo_add(value, value, cryptoContext)
        value = alignment.rescale_one_level(value, cryptoContext)
        value = homo.homo_add_scalar_double(value, -1.0, cryptoContext)
        T2.append(value)
    return T2


def _chebyshev_odd_multiple(T2, cryptoContext):
    # Computes T_{k * (2^m - 1)}(x) from T_k, T_{2k}, ...
    value = T2[0]
    for doubled in T2[1:]:
        prod = homo.homo_mul(value, doubled, cryptoContext)
        value = homo.homo_add(prod, prod, cryptoContext)
        value = alignment.rescale_one_level(value, cryptoContext)
        value = homo.homo_sub(value, T2[0], cryptoContext)
    return value


def _q_highest_term(node: ChebyshevPSNode, Tk, cryptoContext, *, root, has_tail):
    if root:
        if has_tail:
            return homo.homo_add(Tk, Tk, cryptoContext)
        value = Tk
        for _ in range(1, int(node.divqr_q[-1])):
            value = homo.homo_add(value, Tk, cryptoContext)
        return value

    coefficient = node.divqr_q[-1] + (1.1 if has_tail else 0.0)
    scalar = 2 ** math.floor(math.log2(coefficient))
    return homo.homo_mul_scalar_int(Tk, scalar, cryptoContext)


def _eval_ps_node(node: ChebyshevPSNode, T, T2, cryptoContext, constants, bootstrap_plan, path, *, root):
    c = _chebyshev_tail(
        node.divcs_q,
        T,
        cryptoContext,
        constants,
        bootstrap_plan,
        (*path, "c"),
        identity_shortcut=True,
    )
    if c is not None:
        c = _add_chebyshev_constant(c, node.divcs_q, cryptoContext)
        if not root and cryptoContext.rescaleTech == "FIXEDMANUAL":
            target = T2[node.m - 1]
            c = alignment.align_to(
                c,
                alignment.CipherState(target.cur_limbs, target.noise_deg, target.scaling_factor),
                cryptoContext,
            )

    if node.q_node is not None:
        q = _eval_ps_node(node.q_node, T, T2, cryptoContext, constants, bootstrap_plan, (*path, "q_node"), root=False)
    else:
        q = _chebyshev_tail(
            node.divqr_q,
            T,
            cryptoContext,
            constants,
            bootstrap_plan,
            (*path, "q"),
            size=node.k,
        )
        highest = _q_highest_term(node, T.items[node.k - 1], cryptoContext, root=root, has_tail=q is not None)
        q = highest if q is None else homo.homo_add(q, highest, cryptoContext)
        q = _add_chebyshev_constant(q, node.divqr_q, cryptoContext)

    if node.s_node is not None:
        s = _eval_ps_node(node.s_node, T, T2, cryptoContext, constants, bootstrap_plan, (*path, "s_node"), root=False)
    else:
        s = _chebyshev_tail(
            node.s2,
            T,
            cryptoContext,
            constants,
            bootstrap_plan,
            (*path, "s"),
            size=node.k,
        )
        s = T.items[node.k - 1] if s is None else homo.homo_add(s, T.items[node.k - 1], cryptoContext)
        s = _add_chebyshev_constant(s, node.s2, cryptoContext)
        if not root and cryptoContext.rescaleTech == "FIXEDMANUAL":
            s = alignment.align_to(
                s,
                alignment.CipherState(s.cur_limbs - 1, 1, None),
                cryptoContext,
            )

    base = T2[node.m - 1]
    if c is None:
        result = homo.homo_add_scalar_double(base, node.divcs_q[0] / 2, cryptoContext)
    else:
        result = homo.homo_add(base, c, cryptoContext)

    result = homo.homo_mul(result, q, cryptoContext)
    result = alignment.rescale_one_level(result, cryptoContext)
    result = homo.homo_add(result, s, cryptoContext)

    if root:
        result = homo.homo_sub(result, _chebyshev_odd_multiple(T2, cryptoContext), cryptoContext)
    return result


# note: EvalChebyshevSeriesPS in ckksrns-advancedshe.cpp
def eval_bootstrapping_chebyshev(x, a, b, cryptoContext, constants, bootstrap_plan):
    plan = get_bootstrap_approx_plan(cryptoContext.secretKeyDist)
    root = plan.ps_root
    T = _chebyshev_basis(x, a, b, root.k, cryptoContext)
    T2 = _chebyshev_doubling_basis(T.items[-1], root.m, cryptoContext)
    return _eval_ps_node(root, T, T2, cryptoContext, constants, bootstrap_plan, ("root",), root=True)


def apply_double_angle_iterations(ciphertext, cryptoContext, constants, bootstrap_plan):
    plan = get_bootstrap_approx_plan(cryptoContext.secretKeyDist)

    for j in range(1, plan.double_angle_iterations + 1):
        ciphertext = homo.homo_square(ciphertext, cryptoContext)
        ciphertext = homo.homo_add(ciphertext, ciphertext, cryptoContext)
        ciphertext = _add_named_scalar_double_preserve_noise(
            ciphertext,
            bootstrap_plan.double_angle_scalar_names[j - 1],
            constants,
            cryptoContext,
        )
        ciphertext = alignment.rescale_one_level(ciphertext, cryptoContext)
    return ciphertext


def eval_bootstrap_approx_mod(ciphertext, cryptoContext, constants, bootstrap_plan):
    ciphertext = eval_bootstrapping_chebyshev(ciphertext, -1, 1, cryptoContext, constants, bootstrap_plan)
    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        ciphertext = alignment.rescale_one_level(ciphertext, cryptoContext)
    return apply_double_angle_iterations(ciphertext, cryptoContext, constants, bootstrap_plan)
