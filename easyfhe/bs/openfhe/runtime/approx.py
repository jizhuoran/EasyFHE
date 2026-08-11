import math

from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import layout
from easyfhe.fhe.ciphertext import EncodedScalar

from ..generation.plan import (
    ALIGN_C_TO_BASE,
    ALIGN_S_DROP_TO_NOISE_ONE,
    KIND_C,
    KIND_Q,
    KIND_S,
    Q_HIGHEST_ROOT_DOUBLE,
    Q_HIGHEST_ROOT_REPEAT,
    Q_HIGHEST_SCALAR,
    SPACE_NODE,
    SPACE_SMALL,
)


# ---------------------------------------------------------------------------
# Scalar encoding and state helpers


def _mul_rescale_constant(name, in0, in1, constants, cryptoContext):
    if str(getattr(cryptoContext, "scale_mode", "")).lower() == "flexible":
        target = _mul_alignment_target_for_bs(in0, in1)
        divisor = float(cryptoContext.rescale_divisor_at(target.cur_limbs - 1))
        scalar_scale = float(target.scaling_factor) * float(target.scaling_factor) / divisor
        return constants.encoded_scalars(
            name,
            cur_limbs=target.cur_limbs - 1,
            scale_degree=1,
            context=cryptoContext,
            mode="scaled",
            scaling_factor=scalar_scale,
        )[0]
    target, _ = alignment.plan_mul_alignment(in0, in1, cryptoContext)
    return constants.encoded_scalars(
        name,
        cur_limbs=target.cur_limbs - 1,
        scale_degree=1,
        context=cryptoContext,
        mode="scaled",
    )[0]


def _add_chebyshev_constant(value, scalar_path, constants, bootstrap_plan, cryptoContext):
    return arithmetic.homo_add_scalar(
        value,
        constants.encoded_scalars(
            bootstrap_plan.approx_constant_scalar_names[tuple(scalar_path)],
            cur_limbs=value.state.cur_limbs,
            scale_degree=value.state.scale_degree,
            context=cryptoContext,
            mode="scaled",
            scaling_factor=value.state.scaling_factor,
        )[0],
        cryptoContext,
    )


# ---------------------------------------------------------------------------
# Chebyshev basis construction


def _require_unit_interval(a, b):
    alpha = 2 / (b - a)
    beta = 2 * a / (b - a)
    if not (math.isclose(alpha, 1.0) and math.isclose(beta, -1.0)):
        raise NotImplementedError(
            "OpenFHE bootstrap approx runtime only supports the precomputed interval [-1, 1]"
        )


def _is_flexible(cryptoContext):
    return str(getattr(cryptoContext, "scale_mode", "")).lower() == "flexible"


def _align_mul_operands(left, right, cryptoContext):
    if not _is_flexible(cryptoContext):
        return left, right
    target = _mul_alignment_target_for_bs(left, right)
    return (
        alignment.align_to(left, target, cryptoContext),
        alignment.align_to(right, target, cryptoContext),
    )


def _mul_alignment_target_for_bs(left, right):
    target_limbs = min(int(left.state.cur_limbs), int(right.state.cur_limbs))
    for cipher in (left, right):
        if int(cipher.state.cur_limbs) == target_limbs:
            return alignment.CipherState(target_limbs, cipher.state.scale_degree, cipher.state.scaling_factor)
    return alignment.CipherState(target_limbs, left.state.scale_degree, left.state.scaling_factor)


def _align_add_operands(left, right, cryptoContext):
    if not _is_flexible(cryptoContext):
        return left, right
    if int(left.state.cur_limbs) < int(right.state.cur_limbs):
        target = left.state
    elif int(right.state.cur_limbs) < int(left.state.cur_limbs):
        target = right.state
    elif int(left.state.scale_degree) <= int(right.state.scale_degree):
        target = left.state
    else:
        target = right.state
    return (
        alignment.align_to(left, target, cryptoContext),
        alignment.align_to(right, target, cryptoContext),
    )


def _add_aligned(left, right, cryptoContext):
    left, right = _align_add_operands(left, right, cryptoContext)
    return arithmetic.homo_add(left, right, cryptoContext)


def _sub_aligned(left, right, cryptoContext):
    left, right = _align_add_operands(left, right, cryptoContext)
    return arithmetic.homo_sub(left, right, cryptoContext)


def _chebyshev_basis(unit_x, k, cryptoContext, constants, bootstrap_plan):
    T = [unit_x]
    for order in range(2, k + 1):
        lhs = T[order // 2 - 1]
        rhs = T[(order + 1) // 2 - 1]
        lhs, rhs = _align_mul_operands(lhs, rhs, cryptoContext)
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
        final.state.scale_degree,
        final.state.scaling_factor,
    )
    items = tuple(alignment.align_to(item, target, cryptoContext) for item in T)
    return items, layout.pack_cipher_batch(items)


def _chebyshev_doubling_basis(Tk, m, cryptoContext, constants, bootstrap_plan):
    # T2[i] = T_{k * 2^i}(x). T2[0] is T_k(x).
    T2 = [Tk]
    for _ in range(1, m):
        lhs, rhs = _align_mul_operands(T2[-1], T2[-1], cryptoContext)
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
        T2.append(value)
    return T2


def _chebyshev_odd_multiple(T2, cryptoContext):
    # Computes T_{k * (2^m - 1)}(x) from T_k, T_{2k}, ...
    value = T2[0]
    for doubled in T2[1:]:
        value, doubled = _align_mul_operands(value, doubled, cryptoContext)
        value = arithmetic.homo_mul_relin_rescale_postop(
            value,
            doubled,
            cryptoContext,
            apply_double=True,
            sub=T2[0],
        )
    return value


# ---------------------------------------------------------------------------
# Small PS specs: padded tail table plus C/Q/S finalization


def _batch_prefix(cipher, size):
    size = int(size)
    if size == cipher.batch_size:
        return cipher
    return cipher.cipher_like(
        [component[:size] for component in cipher.cv],
        batch_size=size,
    )


def _tail_scalar_table(flat, batch, constants, cryptoContext, bootstrap_plan):
    names = tuple(name for row in bootstrap_plan.approx_tail_scalar_names for name in row)
    scalars = constants.encoded_scalars(
        names,
        cur_limbs=batch.state.cur_limbs,
        scale_degree=1,
        context=cryptoContext,
        mode="scaled",
        scaling_factor=batch.state.scaling_factor,
    )
    return EncodedScalar(
        scalars.residues.reshape(
            len(flat.tail_specs),
            flat.tail_max_deg,
            batch.state.cur_limbs,
        ).contiguous(),
        scalars.cur_limbs,
        scalars.scale_degree,
        scalars.scaling_factor,
    )


def _eval_tail_table(flat, T_batch, cryptoContext, constants, bootstrap_plan):
    if not flat.tail_specs:
        return ()
    batch = _batch_prefix(T_batch, flat.tail_max_deg)
    scalars = _tail_scalar_table(flat, batch, constants, cryptoContext, bootstrap_plan)
    tails = arithmetic.grouped_scalar_weighted_acc(batch, scalars, cryptoContext)
    tails = alignment.rescale(tails, cryptoContext)
    return tuple(layout.cipher_batch_item(tails, index) for index in range(len(flat.tail_specs)))


def _small_tail(spec, tail_values, T_items):
    if spec.direct_t1:
        return T_items[0]
    if spec.tail_idx is None:
        return None
    return tail_values[spec.tail_idx]


def _q_highest_term(spec, Tk, constants, bootstrap_plan, cryptoContext):
    if spec.q_highest_mode == Q_HIGHEST_ROOT_DOUBLE:
        return arithmetic.homo_add(Tk, Tk, cryptoContext)

    if spec.q_highest_mode == Q_HIGHEST_ROOT_REPEAT:
        value = Tk
        for _ in range(1, int(spec.q_highest_repeat)):
            value = arithmetic.homo_add(value, Tk, cryptoContext)
        return value

    if spec.q_highest_mode == Q_HIGHEST_SCALAR:
        return arithmetic.homo_mul_scalar(
            Tk,
            constants.encoded_scalars(
                bootstrap_plan.approx_q_highest_scalar_names[spec.scalar_path],
                cur_limbs=Tk.state.cur_limbs,
                scale_degree=0,
                context=cryptoContext,
                mode="integer",
            )[0],
            cryptoContext,
        )

    raise ValueError(f"unexpected Q highest mode: {spec.q_highest_mode}")


def _finish_c_spec(spec, tail, T2, constants, bootstrap_plan, cryptoContext):
    if tail is None:
        return None

    value = _add_chebyshev_constant(
        tail,
        spec.scalar_path,
        constants,
        bootstrap_plan,
        cryptoContext,
    )
    if spec.align_policy == ALIGN_C_TO_BASE:
        target = T2[spec.m - 1]
        value = alignment.align_to(
            value,
            alignment.CipherState(
                target.state.cur_limbs,
                target.state.scale_degree,
                target.state.scaling_factor,
            ),
            cryptoContext,
        )
    return value


def _finish_q_spec(spec, tail, T_items, constants, bootstrap_plan, cryptoContext):
    highest = _q_highest_term(spec, T_items[spec.k - 1], constants, bootstrap_plan, cryptoContext)
    value = highest if tail is None else _add_aligned(tail, highest, cryptoContext)
    return _add_chebyshev_constant(
        value,
        spec.scalar_path,
        constants,
        bootstrap_plan,
        cryptoContext,
    )


def _finish_s_spec(spec, tail, T_items, constants, bootstrap_plan, cryptoContext):
    Tk = T_items[spec.k - 1]
    value = Tk if tail is None else _add_aligned(tail, Tk, cryptoContext)
    value = _add_chebyshev_constant(
        value,
        spec.scalar_path,
        constants,
        bootstrap_plan,
        cryptoContext,
    )
    if spec.align_policy == ALIGN_S_DROP_TO_NOISE_ONE:
        divisor = float(cryptoContext.rescale_divisor_at(value.state.cur_limbs - 1))
        source_scale = None if value.state.scaling_factor is None else float(value.state.scaling_factor)
        if source_scale is None:
            target_scale = None
        elif int(value.state.scale_degree) > 1:
            target_scale = source_scale / divisor
        else:
            target_scale = source_scale * source_scale / divisor
        value = alignment.align_to(
            value,
            alignment.CipherState(
                value.state.cur_limbs - 1,
                1,
                target_scale,
            ),
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
    tail_values = _eval_tail_table(flat, T_batch, cryptoContext, constants, bootstrap_plan)

    small_values = [None] * len(flat.small_specs)
    for spec in flat.small_specs:
        small_values[spec.out_idx] = _finish_small_spec(
            spec,
            _small_tail(spec, tail_values, T_items),
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
        left = arithmetic.homo_add_scalar(
            base,
            constants.encoded_scalars(
                bootstrap_plan.approx_constant_scalar_names[spec.c_const_scalar_path],
                cur_limbs=base.state.cur_limbs,
                scale_degree=base.state.scale_degree,
                context=cryptoContext,
                mode="scaled",
                scaling_factor=base.state.scaling_factor,
            )[0],
            cryptoContext,
        )
    else:
        left = _add_aligned(base, c, cryptoContext)

    left, q = _align_mul_operands(left, q, cryptoContext)
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

    _require_unit_interval(a, b)
    T_items, T_batch = _chebyshev_basis(x, flat.k, cryptoContext, constants, bootstrap_plan)
    T2 = _chebyshev_doubling_basis(T_items[-1], flat.m, cryptoContext, constants, bootstrap_plan)

    small_values = _eval_small_specs(flat, T_items, T_batch, T2, cryptoContext, constants, bootstrap_plan)
    node_values = _eval_combine_specs(flat, small_values, T2, constants, bootstrap_plan, cryptoContext)

    result = _read_ref(flat.root_ref, small_values, node_values)
    return _sub_aligned(result, _chebyshev_odd_multiple(T2, cryptoContext), cryptoContext)


def apply_double_angle_iterations(ciphertext, cryptoContext, constants, bootstrap_plan):
    for j in range(bootstrap_plan.double_angle_iterations):
        squared = arithmetic.homo_mul_relin(
            ciphertext,
            ciphertext,
            cryptoContext,
        )
        doubled = arithmetic._cipher_add(squared, squared, cryptoContext)
        scalar = constants.encoded_scalars(
            bootstrap_plan.double_angle_scalar_names[j],
            cur_limbs=doubled.state.cur_limbs,
            scale_degree=doubled.state.scale_degree,
            context=cryptoContext,
            mode="scaled",
            scaling_factor=doubled.state.scaling_factor,
        )[0]
        ciphertext = arithmetic.homo_add_scalar(doubled, scalar, cryptoContext)
        ciphertext = alignment.rescale(ciphertext, cryptoContext)
    return ciphertext


def eval_bootstrap_approx_mod(ciphertext, cryptoContext, constants, bootstrap_plan):
    ciphertext = eval_bootstrapping_chebyshev(ciphertext, -1, 1, cryptoContext, constants, bootstrap_plan)
    return apply_double_angle_iterations(ciphertext, cryptoContext, constants, bootstrap_plan)
