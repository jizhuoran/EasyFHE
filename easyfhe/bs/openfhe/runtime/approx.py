import math
from dataclasses import dataclass

import numpy as np
import easyfhe as torch

from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import layout

from ..generation.plan import (
    KIND_C,
    KIND_Q,
    KIND_S,
    SPACE_NODE,
    SPACE_SMALL,
    ChebyshevPSNode,
    degree,
    get_bootstrap_approx_plan,
)


@dataclass(frozen=True)
class _SmallTailRequest:
    spec: object
    deg: int
    scalar_path: tuple


# ---------------------------------------------------------------------------
# Scalar encoding and state helpers


def _uses_manual_rescale(cryptoContext):
    return cryptoContext.scale_mode == "fixed" and cryptoContext.rescale_policy == "manual"


def _mul_rescale_constant(name, in0, in1, constants, cryptoContext):
    target, _ = alignment.plan_mul_alignment(in0, in1, cryptoContext)
    return constants.encoded_scalars(
        name,
        target.cur_limbs - 1,
        1,
        cryptoContext,
        mode="double",
    )[0]


def _add_chebyshev_constant(value, scalar_path, constants, bootstrap_plan, cryptoContext):
    return arithmetic.homo_add_scalar_double(
        value,
        constants.encoded_scalars(
            bootstrap_plan.approx_constant_scalar_names[tuple(scalar_path)],
            value.state.cur_limbs,
            1,
            cryptoContext,
            mode="double",
        )[0],
        cryptoContext,
    )


# ---------------------------------------------------------------------------
# Chebyshev basis construction


def _scale_input_to_unit_interval(x, a, b, cryptoContext):
    alpha = 2 / (b - a)
    beta = 2 * a / (b - a)
    if not (math.isclose(alpha, 1.0) and math.isclose(beta, -1.0)):
        raise NotImplementedError(
            "OpenFHE bootstrap approx runtime only supports the precomputed interval [-1, 1]"
        )
    return x


def _chebyshev_basis(unit_x, k, cryptoContext, constants, bootstrap_plan):
    T = [unit_x]
    for order in range(2, k + 1):
        lhs = T[order // 2 - 1]
        rhs = T[(order + 1) // 2 - 1]
        if order & 1:
            value = arithmetic.homo_mul_relin_rescale_postop(
                lhs,
                rhs,
                cryptoContext,
                apply_double=True,
                sub=T[0],
            )
        else:
            value = arithmetic.homo_mul_relin_rescale_postop(
                lhs,
                rhs,
                cryptoContext,
                apply_double=True,
                scalar=_mul_rescale_constant(
                    bootstrap_plan.chebyshev_neg_one_scalar_name,
                    lhs,
                    rhs,
                    constants,
                    cryptoContext,
                ),
            )
        T.append(value)

    final = T[-1]
    target = alignment.CipherState(
        final.state.cur_limbs,
        final.state.noise_deg,
        final.state.scaling_factor,
    )
    items = tuple(alignment.align_to(item, target, cryptoContext) for item in T)
    return items, layout.pack_cipher_batch(items)


def _chebyshev_doubling_basis(Tk, m, cryptoContext, constants, bootstrap_plan):
    # T2[i] = T_{k * 2^i}(x). T2[0] is T_k(x).
    T2 = [Tk]
    for _ in range(1, m):
        value = arithmetic.homo_mul_relin_rescale_postop(
            T2[-1],
            T2[-1],
            cryptoContext,
            apply_double=True,
            scalar=_mul_rescale_constant(
                bootstrap_plan.chebyshev_neg_one_scalar_name,
                T2[-1],
                T2[-1],
                constants,
                cryptoContext,
            ),
        )
        T2.append(value)
    return T2


def _chebyshev_odd_multiple(T2, cryptoContext):
    # Computes T_{k * (2^m - 1)}(x) from T_k, T_{2k}, ...
    value = T2[0]
    for doubled in T2[1:]:
        value = arithmetic.homo_mul_relin_rescale_postop(
            value,
            doubled,
            cryptoContext,
            apply_double=True,
            sub=T2[0],
        )
    return value


# ---------------------------------------------------------------------------
# Small PS specs: C/Q/S tails over T1..Tk


def _batch_prefix(cipher, size):
    size = int(size)
    if size == cipher.batch_size:
        return cipher
    return cipher.cipher_like(
        [component[:size] for component in cipher.cv],
        batch_size=size,
    )


def _truncated_degree(coefficients, size):
    truncated = np.copy(coefficients[:size])
    truncated.resize(size, refcheck=False)
    return degree(truncated)


def _small_coefficients_and_path(spec):
    node = spec.node
    if spec.kind == KIND_C:
        return node.divcs_q, (*spec.path, "c"), None
    if spec.kind == KIND_Q:
        return node.divqr_q, (*spec.path, "q"), node.k
    if spec.kind == KIND_S:
        return node.s2, (*spec.path, "s"), node.k
    raise ValueError(f"unknown flat PS small spec kind: {spec.kind}")


def _make_small_tail_request(spec, T_items, bootstrap_plan):
    coefficients, scalar_path, size = _small_coefficients_and_path(spec)
    deg = degree(coefficients) if size is None else _truncated_degree(coefficients, size)
    if deg < 1:
        return None, None
    if spec.kind == KIND_C and deg == 1 and coefficients[1] == 1:
        return None, T_items[0]

    scalar_path = tuple(scalar_path)
    names = bootstrap_plan.approx_scalar_names[scalar_path]
    if len(names) != int(deg):
        raise ValueError(
            f"Chebyshev scalar metadata mismatch at {scalar_path}: "
            f"got {len(names)}, expected {deg}"
        )
    return _SmallTailRequest(spec=spec, deg=int(deg), scalar_path=scalar_path), None


def _collect_small_tail_requests(flat, T_items, bootstrap_plan):
    tail_values = [None] * len(flat.small_specs)
    grouped_requests = {}

    for spec in flat.small_specs:
        request, direct_tail = _make_small_tail_request(spec, T_items, bootstrap_plan)
        if request is None:
            tail_values[spec.out_idx] = direct_tail
        else:
            grouped_requests.setdefault(request.deg, []).append(request)

    return tail_values, grouped_requests


def _tail_scalars_for_requests(requests, batch, constants, cryptoContext, bootstrap_plan):
    rows = []
    for request in requests:
        names = bootstrap_plan.approx_scalar_names[request.scalar_path]
        rows.append(
            constants.encoded_scalars(
                names,
                batch.state.cur_limbs,
                1,
                cryptoContext,
                mode="double",
            )
        )
    return torch.stack(rows, dim=0).contiguous()


def _normalize_grouped_tail_shape(tails, count, cryptoContext):
    if tails.batch_size <= 1 or tails.cv[0].dim() != 2:
        return tails
    return tails.cipher_like(
        [
            component.reshape(count, tails.state.cur_limbs, cryptoContext.N)
            for component in tails.cv
        ],
        batch_size=count,
    )


def _eval_grouped_tail_requests(tail_values, grouped_requests, T_batch, cryptoContext, constants, bootstrap_plan):
    for deg, requests in grouped_requests.items():
        batch = _batch_prefix(T_batch, deg)
        scalars = _tail_scalars_for_requests(
            requests,
            batch,
            constants,
            cryptoContext,
            bootstrap_plan,
        )
        tails = arithmetic.grouped_scalar_weighted_acc(batch, scalars, cryptoContext)
        tails = alignment.rescale_one_level(tails, cryptoContext)
        tails = _normalize_grouped_tail_shape(tails, len(requests), cryptoContext)

        for index, request in enumerate(requests):
            tail_values[request.spec.out_idx] = layout.cipher_batch_item(tails, index)


def _q_highest_term(node: ChebyshevPSNode, path, Tk, constants, bootstrap_plan, cryptoContext, *, root, has_tail):
    if root:
        if has_tail:
            return arithmetic.homo_add(Tk, Tk, cryptoContext)
        value = Tk
        for _ in range(1, int(node.divqr_q[-1])):
            value = arithmetic.homo_add(value, Tk, cryptoContext)
        return value

    return arithmetic.homo_mul_scalar_int(
        Tk,
        constants.encoded_scalars(
            bootstrap_plan.approx_q_highest_scalar_names[tuple((*path, "q"))],
            Tk.state.cur_limbs,
            0,
            cryptoContext,
            mode="int",
        )[0],
        cryptoContext,
    )


def _finish_c_spec(spec, tail, T2, constants, bootstrap_plan, cryptoContext):
    if tail is None:
        return None

    value = _add_chebyshev_constant(
        tail,
        (*spec.path, "c"),
        constants,
        bootstrap_plan,
        cryptoContext,
    )
    if not spec.root and _uses_manual_rescale(cryptoContext):
        target = T2[spec.node.m - 1]
        value = alignment.align_to(
            value,
            alignment.CipherState(
                target.state.cur_limbs,
                target.state.noise_deg,
                target.state.scaling_factor,
            ),
            cryptoContext,
        )
    return value


def _finish_q_spec(spec, tail, T_items, constants, bootstrap_plan, cryptoContext):
    node = spec.node
    highest = _q_highest_term(
        node,
        spec.path,
        T_items[node.k - 1],
        constants,
        bootstrap_plan,
        cryptoContext,
        root=spec.root,
        has_tail=tail is not None,
    )
    value = highest if tail is None else arithmetic.homo_add(tail, highest, cryptoContext)
    return _add_chebyshev_constant(
        value,
        (*spec.path, "q"),
        constants,
        bootstrap_plan,
        cryptoContext,
    )


def _finish_s_spec(spec, tail, T_items, constants, bootstrap_plan, cryptoContext):
    node = spec.node
    Tk = T_items[node.k - 1]
    value = Tk if tail is None else arithmetic.homo_add(tail, Tk, cryptoContext)
    value = _add_chebyshev_constant(
        value,
        (*spec.path, "s"),
        constants,
        bootstrap_plan,
        cryptoContext,
    )
    if not spec.root and _uses_manual_rescale(cryptoContext):
        value = alignment.align_to(
            value,
            alignment.CipherState(value.state.cur_limbs - 1, 1, None),
            cryptoContext,
        )
    return value


def _finish_small_spec(spec, tail, T_items, T2, constants, bootstrap_plan, cryptoContext):
    if spec.kind == KIND_C:
        return _finish_c_spec(spec, tail, T2, constants, bootstrap_plan, cryptoContext)
    if spec.kind == KIND_Q:
        return _finish_q_spec(spec, tail, T_items, constants, bootstrap_plan, cryptoContext)
    if spec.kind == KIND_S:
        return _finish_s_spec(spec, tail, T_items, constants, bootstrap_plan, cryptoContext)
    raise ValueError(f"unknown flat PS small spec kind: {spec.kind}")


def _eval_small_specs(flat, T_items, T_batch, T2, cryptoContext, constants, bootstrap_plan):
    tail_values, grouped_requests = _collect_small_tail_requests(flat, T_items, bootstrap_plan)
    _eval_grouped_tail_requests(
        tail_values,
        grouped_requests,
        T_batch,
        cryptoContext,
        constants,
        bootstrap_plan,
    )

    small_values = [None] * len(flat.small_specs)
    for spec in flat.small_specs:
        small_values[spec.out_idx] = _finish_small_spec(
            spec,
            tail_values[spec.out_idx],
            T_items,
            T2,
            constants,
            bootstrap_plan,
            cryptoContext,
        )
    return small_values


# ---------------------------------------------------------------------------
# Giant PS combine specs


def _read_ref(ref, small_values, node_values):
    space, idx = ref
    if space == SPACE_SMALL:
        return small_values[idx]
    if space == SPACE_NODE:
        return node_values[idx]
    raise ValueError(f"unknown flat PS value space: {space}")


def _eval_combine_spec(spec, small_values, node_values, T2, constants, bootstrap_plan, cryptoContext):
    c = _read_ref(spec.c_ref, small_values, node_values)
    q = _read_ref(spec.q_ref, small_values, node_values)
    s = _read_ref(spec.s_ref, small_values, node_values)
    base = T2[spec.base_idx]

    if c is None:
        left = arithmetic.homo_add_scalar_double(
            base,
            constants.encoded_scalars(
                bootstrap_plan.approx_constant_scalar_names[tuple((*spec.path, "c"))],
                base.state.cur_limbs,
                1,
                cryptoContext,
                mode="double",
            )[0],
            cryptoContext,
        )
    else:
        left = arithmetic.homo_add(base, c, cryptoContext)

    return arithmetic.homo_mul_relin_rescale_postop(left, q, cryptoContext, apply_double=False, add=s)


def _eval_combine_specs(flat, small_values, T2, constants, bootstrap_plan, cryptoContext):
    node_values = [None] * flat.node_count
    for spec in flat.combine_specs:
        node_values[spec.out_idx] = _eval_combine_spec(
            spec,
            small_values,
            node_values,
            T2,
            constants,
            bootstrap_plan,
            cryptoContext,
        )
    return node_values


# ---------------------------------------------------------------------------
# Runtime entry points


def eval_bootstrapping_chebyshev(x, a, b, cryptoContext, constants, bootstrap_plan):
    flat = bootstrap_plan.approx_eval_plan

    unit_x = _scale_input_to_unit_interval(x, a, b, cryptoContext)
    T_items, T_batch = _chebyshev_basis(unit_x, flat.k, cryptoContext, constants, bootstrap_plan)
    T2 = _chebyshev_doubling_basis(T_items[-1], flat.m, cryptoContext, constants, bootstrap_plan)

    small_values = _eval_small_specs(flat, T_items, T_batch, T2, cryptoContext, constants, bootstrap_plan)
    node_values = _eval_combine_specs(flat, small_values, T2, constants, bootstrap_plan, cryptoContext)

    result = _read_ref(flat.root_ref, small_values, node_values)
    return arithmetic.homo_sub(result, _chebyshev_odd_multiple(T2, cryptoContext), cryptoContext)


def apply_double_angle_iterations(ciphertext, cryptoContext, constants, bootstrap_plan):
    plan = get_bootstrap_approx_plan(cryptoContext.secretKeyDist)

    for j in range(1, plan.double_angle_iterations + 1):
        ciphertext = arithmetic.homo_mul_relin_rescale_postop(
            ciphertext,
            ciphertext,
            cryptoContext,
            apply_double=True,
            scalar=_mul_rescale_constant(
                bootstrap_plan.double_angle_scalar_names[j - 1],
                ciphertext,
                ciphertext,
                constants,
                cryptoContext,
            ),
        )
    return ciphertext


def eval_bootstrap_approx_mod(ciphertext, cryptoContext, constants, bootstrap_plan):
    ciphertext = eval_bootstrapping_chebyshev(ciphertext, -1, 1, cryptoContext, constants, bootstrap_plan)
    if not _uses_manual_rescale(cryptoContext) and ciphertext.state.noise_deg > 1:
        ciphertext = alignment.rescale_one_level(ciphertext, cryptoContext)
    return apply_double_angle_iterations(ciphertext, cryptoContext, constants, bootstrap_plan)
