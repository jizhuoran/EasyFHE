import math
from dataclasses import dataclass

import numpy as np
import easyfhe as torch

from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import layout
from easyfhe.fhe.ops import rotation
from easyfhe.fhe.ops.primitives import _cipher_add_scalar, _cipher_sub_scalar

from .approx_plan import (
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
class _ChebyshevBasis:
    items: tuple
    batch: object


@dataclass(frozen=True)
class _ChebyshevTailRequest:
    spec: object
    deg: int
    scalar_path: tuple


def _add_named_scalar_double_preserve_noise(ciphertext, name, constants, cryptoContext):
    constant = constants._scalar_value(name)
    encoded = constants.encoded_scalars(
        name,
        ciphertext.state.cur_limbs,
        ciphertext.state.noise_deg,
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
    )


def _grouped_scalar_weighted_acc(cipher_batch, scalars, cryptoContext):
    return arithmetic.grouped_scalar_weighted_acc(cipher_batch, scalars, cryptoContext)


def _truncated_degree(coefficients, size):
    truncated = np.copy(coefficients[:size])
    truncated.resize(size, refcheck=False)
    return degree(truncated)


def _small_tail_metadata(spec):
    node = spec.node
    if spec.kind == KIND_C:
        return node.divcs_q, (*spec.path, "c"), None, True
    if spec.kind == KIND_Q:
        return node.divqr_q, (*spec.path, "q"), node.k, False
    if spec.kind == KIND_S:
        return node.s2, (*spec.path, "s"), node.k, False
    raise ValueError(f"unknown flat PS small spec kind: {spec.kind}")


def _make_chebyshev_tail_request(spec, T, bootstrap_plan):
    coefficients, scalar_path, size, identity_shortcut = _small_tail_metadata(spec)
    deg = degree(coefficients) if size is None else _truncated_degree(coefficients, size)
    if deg < 1:
        return None, None
    if identity_shortcut and deg == 1 and coefficients[1] == 1:
        return None, T.items[0]

    names = bootstrap_plan.approx_scalar_names[tuple(scalar_path)]
    if len(names) != int(deg):
        raise ValueError(
            f"Chebyshev scalar metadata mismatch at {scalar_path}: "
            f"got {len(names)}, expected {deg}"
        )
    return _ChebyshevTailRequest(
        spec=spec,
        deg=int(deg),
        scalar_path=tuple(scalar_path),
    ), None


def _eval_small_spec_from_tail(spec, tail, T, T2, cryptoContext):
    node = spec.node
    if spec.kind == KIND_C:
        c = tail
        if c is not None:
            c = _add_chebyshev_constant(c, node.divcs_q, cryptoContext)
            if not spec.root:
                target = T2[node.m - 1]
                c = alignment.align_to(
                    c,
                    alignment.CipherState(
                        target.state.cur_limbs,
                        target.state.noise_deg,
                        target.state.scaling_factor,
                    ),
                    cryptoContext,
                )
        return c

    if spec.kind == KIND_Q:
        q = tail
        highest = _q_highest_term(
            node,
            T.items[node.k - 1],
            cryptoContext,
            root=spec.root,
            has_tail=q is not None,
        )
        q = highest if q is None else arithmetic.homo_add(q, highest, cryptoContext)
        return _add_chebyshev_constant(q, node.divqr_q, cryptoContext)

    if spec.kind == KIND_S:
        s = tail
        if s is None:
            s = T.items[node.k - 1]
        else:
            s = arithmetic.homo_add(s, T.items[node.k - 1], cryptoContext)
        s = _add_chebyshev_constant(s, node.s2, cryptoContext)
        if not spec.root:
            s = alignment.align_to(
                s,
                alignment.CipherState(s.state.cur_limbs - 1, 1, None),
                cryptoContext,
            )
        return s

    raise ValueError(f"unknown flat PS small spec kind: {spec.kind}")


def _eval_small_specs_grouped(flat, T, T2, cryptoContext, constants, bootstrap_plan):
    tail_values = [None] * len(flat.small_specs)
    grouped_requests = {}

    for spec in flat.small_specs:
        request, direct_tail = _make_chebyshev_tail_request(spec, T, bootstrap_plan)
        if request is None:
            tail_values[spec.out_idx] = direct_tail
        else:
            grouped_requests.setdefault(request.deg, []).append(request)

    for deg, requests in grouped_requests.items():
        batch = _batch_prefix(T.batch, deg)
        scalar_rows = []
        for request in requests:
            names = bootstrap_plan.approx_scalar_names[request.scalar_path]
            scalar_rows.append(
                constants.encoded_scalars(
                    names,
                    batch.state.cur_limbs,
                    1,
                    cryptoContext,
                    mode="double",
                )
            )
        scalars = torch.stack(scalar_rows, dim=0).contiguous()
        tails = alignment.rescale_one_level(
            _grouped_scalar_weighted_acc(batch, scalars, cryptoContext),
            cryptoContext,
        )
        if tails.batch_size > 1 and tails.cv[0].dim() == 2:
            tails = tails.cipher_like(
                [
                    component.reshape(len(requests), tails.state.cur_limbs, cryptoContext.N)
                    for component in tails.cv
                ],
                batch_size=len(requests),
            )
        for index, request in enumerate(requests):
            tail_values[request.spec.out_idx] = layout.cipher_batch_item(tails, index)

    small_values = [None] * len(flat.small_specs)
    for spec in flat.small_specs:
        small_values[spec.out_idx] = _eval_small_spec_from_tail(
            spec,
            tail_values[spec.out_idx],
            T,
            T2,
            cryptoContext,
        )
    return small_values


def _add_chebyshev_constant(value, coefficients, cryptoContext):
    return arithmetic.homo_add_scalar_double(
        value,
        _encoded_double_scalar(coefficients[0] / 2, value.state.cur_limbs, cryptoContext),
        cryptoContext,
    )


def _encoded_double_scalar(value, cur_limbs, cryptoContext):
    return arithmetic._encode_double_for_scalar_op(value, cur_limbs, cryptoContext)


def _encoded_int_scalar(value, cur_limbs, cryptoContext):
    return arithmetic._encode_int_for_scalar_op(value, cur_limbs, cryptoContext)


def _encoded_mul_rescale_scalar(in0, in1, value, cryptoContext):
    target, _ = alignment.plan_mul_alignment(in0, in1, cryptoContext)
    return _encoded_double_scalar(value, target.cur_limbs - 1, cryptoContext)


def _chebyshev_basis(x, a, b, k, cryptoContext):
    T = [_scale_input_to_unit_interval(x, a, b, cryptoContext)]
    for order in range(2, k + 1):
        if order & 1:
            value = arithmetic.homo_mul_relin_rescale_postop(
                T[order // 2 - 1],
                T[(order + 1) // 2 - 1],
                cryptoContext,
                apply_double=True,
                sub=T[0],
            )
        else:
            value = arithmetic.homo_mul_relin_rescale_postop(
                T[order // 2 - 1],
                T[(order + 1) // 2 - 1],
                cryptoContext,
                apply_double=True,
                scalar=_encoded_mul_rescale_scalar(
                    T[order // 2 - 1],
                    T[(order + 1) // 2 - 1],
                    -1.0,
                    cryptoContext,
                ),
            )
        T.append(value)

    final = T[-1]
    # min_limbs = min(item.state.cur_limbs for item in T)
    # if final.state.cur_limbs != min_limbs:
    #     raise ValueError(
    #         f"Chebyshev basis target is not the lowest limb state: "
    #         f"final={final.state.cur_limbs}, min={min_limbs}"
    #     )
    # if any(item.state.noise_deg != 1 for item in T):
    #     raise ValueError(
    #         "Chebyshev basis expects all terms to have noise_deg=1 before packing: "
    #         f"{[item.state.noise_deg for item in T]}"
    #     )
    items = tuple(
        alignment.align_to(
            item,
            alignment.CipherState(final.state.cur_limbs, final.state.noise_deg, final.state.scaling_factor),
            cryptoContext,
        )
        for item in T
    )
    return _ChebyshevBasis(items=items, batch=layout.pack_cipher_batch(items))


def _scale_input_to_unit_interval(x, a, b, cryptoContext):
    y = x
    alpha = 2 / (b - a)
    if not math.isclose(alpha, 1.0):
        y = arithmetic.homo_mul_scalar_double(
            x,
            _encoded_double_scalar(alpha, x.state.cur_limbs, cryptoContext),
            cryptoContext,
        )
        y = alignment.rescale_one_level(y, cryptoContext)

    beta = 2 * a / (b - a)
    if not math.isclose(beta, -1.0):
        y = arithmetic.homo_add_scalar_double(
            y,
            _encoded_double_scalar(-1.0 - beta, y.state.cur_limbs, cryptoContext),
            cryptoContext,
        )
    return y


def _chebyshev_doubling_basis(Tk, m, cryptoContext):
    # T2[i] = T_{k * 2^i}(x). T2[0] is T_k(x).
    T2 = [Tk]
    for _ in range(1, m):
        value = arithmetic.homo_mul_relin_rescale_postop(
            T2[-1],
            T2[-1],
            cryptoContext,
            apply_double=True,
            scalar=_encoded_mul_rescale_scalar(T2[-1], T2[-1], -1.0, cryptoContext),
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


def _q_highest_term(node: ChebyshevPSNode, Tk, cryptoContext, *, root, has_tail):
    if root:
        if has_tail:
            return arithmetic.homo_add(Tk, Tk, cryptoContext)
        value = Tk
        for _ in range(1, int(node.divqr_q[-1])):
            value = arithmetic.homo_add(value, Tk, cryptoContext)
        return value

    coefficient = node.divqr_q[-1] + (1.1 if has_tail else 0.0)
    scalar = 2 ** math.floor(math.log2(coefficient))
    return arithmetic.homo_mul_scalar_int(
        Tk,
        _encoded_int_scalar(scalar, Tk.state.cur_limbs, cryptoContext),
        cryptoContext,
    )


def _read_ref(ref, small_values, node_values):
    space, idx = ref
    if space == SPACE_SMALL:
        return small_values[idx]
    if space == SPACE_NODE:
        return node_values[idx]
    raise ValueError(f"unknown flat PS value space: {space}")


def _eval_combine_spec(spec, small_values, node_values, T2, cryptoContext):
    c = _read_ref(spec.c_ref, small_values, node_values)
    q = _read_ref(spec.q_ref, small_values, node_values)
    s = _read_ref(spec.s_ref, small_values, node_values)
    base = T2[spec.base_idx]

    if c is None:
        result = arithmetic.homo_add_scalar_double(
            base,
            _encoded_double_scalar(spec.node.divcs_q[0] / 2, base.state.cur_limbs, cryptoContext),
            cryptoContext,
        )
    else:
        result = arithmetic.homo_add(base, c, cryptoContext)

    return arithmetic.homo_mul_relin_rescale_postop(result, q, cryptoContext, apply_double=False, add=s)


# note: EvalChebyshevSeriesPS in ckksrns-advancedshe.cpp
def eval_bootstrapping_chebyshev(x, a, b, cryptoContext, constants, bootstrap_plan):
    plan = get_bootstrap_approx_plan(cryptoContext.secretKeyDist)
    if plan.pre_chebyshev_offset:
        x = arithmetic.homo_add_scalar_double(
            x,
            _encoded_double_scalar(plan.pre_chebyshev_offset, x.state.cur_limbs, cryptoContext),
            cryptoContext,
        )
    flat = bootstrap_plan.approx_eval_plan

    T = _chebyshev_basis(x, a, b, flat.k, cryptoContext)
    T2 = _chebyshev_doubling_basis(T.items[-1], flat.m, cryptoContext)

    small_values = _eval_small_specs_grouped(
        flat,
        T,
        T2,
        cryptoContext,
        constants,
        bootstrap_plan,
    )

    node_values = [None] * flat.node_count
    for spec in flat.combine_specs:
        node_values[spec.out_idx] = _eval_combine_spec(
            spec,
            small_values,
            node_values,
            T2,
            cryptoContext,
        )

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
            scalar=_encoded_mul_rescale_scalar(
                ciphertext,
                ciphertext,
                constants._scalar_value(bootstrap_plan.double_angle_scalar_names[j - 1]),
                cryptoContext,
            ),
        )
    return ciphertext


def eval_bootstrap_approx_mod(ciphertext, cryptoContext, constants, bootstrap_plan):
    ciphertext = eval_bootstrapping_chebyshev(ciphertext, -1, 1, cryptoContext, constants, bootstrap_plan)
    return apply_double_angle_iterations(ciphertext, cryptoContext, constants, bootstrap_plan)
