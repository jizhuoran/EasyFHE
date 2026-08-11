from . import kernels as F
from ..ciphertext import Cipher, CipherState, EncodedScalar, Plaintext
from . import validation
from . import alignment
from .primitives import (
    _cipher_add,
    _cipher_add_inplace,
    _cipher_add_plain,
    _cipher_add_plain_inplace,
    _cipher_add_scalar,
    _cipher_add_scalar_inplace,
    _cipher_mul,
    _cipher_mul_plain,
    _cipher_mul_plain_inplace,
    _cipher_mul_scalar_double,
    _cipher_mul_scalar_double_inplace,
    _cipher_mul_scalar_int,
    _cipher_mul_scalar_int_inplace,
    _cipher_sub,
    _cipher_sub_inplace,
    _cipher_sub_scalar,
    _cipher_sub_scalar_inplace,
)


_POST_NONE = 0
_POST_ADD = 1
_POST_SUB = 2
_POST_SCALAR = 3
_POST_PLAINTEXT = 4


def _align_for_add_or_sub(in0, in1, context):
    if _is_flexible(context):
        validation.validate_binary_cipher_op(
            "flexible add/sub",
            in0,
            in1,
            require_same_metadata=("slots", "cur_limbs", "scale_degree", "scaling_factor"),
        )
        return in0, in1
    target = alignment.plan_add_alignment(in0, in1, context)
    return alignment.align_to(in0, target, context), alignment.align_to(in1, target, context)


def _align_for_mul(ct1: Cipher, ct2: Cipher, context):
    if _is_flexible(context):
        _validate_flexible_mul_inputs("flexible cipher multiplication", ct1, ct2)
        return ct1, ct2
    target1, target2 = alignment.plan_mul_alignment(ct1, ct2, context)
    return alignment.align_to(ct1, target1, context), alignment.align_to(ct2, target2, context)


def _is_flexible(context):
    return str(getattr(context, "scale_mode", "")).lower() == "flexible"


def _validate_flexible_mul_inputs(op_name, left, right):
    validation.validate_binary_cipher_op(
        op_name,
        left,
        right,
        require_same_metadata=("slots", "cur_limbs"),
    )
    for arg_name, value in (("left", left), ("right", right)):
        if int(value.state.scale_degree) != 1:
            raise ValueError(
                f"{op_name}: flexible multiplication requires normalized scale_degree=1 inputs; "
                f"{arg_name} has scale_degree={value.state.scale_degree}"
            )
        if value.state.scaling_factor is None:
            raise ValueError(f"{op_name}: {arg_name} scaling_factor is required")


def _validate_flexible_mul_plain_inputs(op_name, cipher, plaintext):
    validation.validate_cipher_plain_op(
        op_name,
        cipher,
        plaintext,
        require_same_metadata=("cur_limbs", "slots"),
    )
    for arg_name, value in (("cipher", cipher), ("plaintext", plaintext)):
        if int(value.state.scale_degree) != 1:
            raise ValueError(
                f"{op_name}: flexible plaintext multiplication requires normalized scale_degree=1 inputs; "
                f"{arg_name} has scale_degree={value.state.scale_degree}"
            )
        if value.state.scaling_factor is None:
            raise ValueError(f"{op_name}: {arg_name} scaling_factor is required")


def _validate_inplace_add_or_sub(op_name, in0, in1):
    validation.validate_binary_cipher_op(
        op_name,
        in0,
        in1,
        require_same_metadata=("slots", "cur_limbs", "scale_degree", "scaling_factor"),
    )


def _require_encoded_scalar(value, op_name):
    if not isinstance(value, EncodedScalar):
        raise TypeError(f"{op_name}: expected EncodedScalar, got {type(value)}")
    return value


def _validate_scalar_limbs(op_name, cipher, scalar):
    if int(scalar.cur_limbs) != int(cipher.state.cur_limbs):
        raise ValueError(
            f"{op_name}: scalar cur_limbs mismatch: "
            f"{scalar.cur_limbs} != {cipher.state.cur_limbs}"
        )


def _validate_scalar_add_sub(op_name, cipher, scalar):
    validation.validate_cipher_op(op_name, cipher, require_ext=False)
    _validate_scalar_limbs(op_name, cipher, scalar)
    validation.validate_matching_metadata(
        op_name,
        cipher,
        _ScalarMetadataView(scalar),
        ("scale_degree", "scaling_factor"),
    )


def _validate_scalar_mul(op_name, cipher, scalar):
    validation.validate_cipher_op(op_name, cipher, require_ext=False)
    _validate_scalar_limbs(op_name, cipher, scalar)
    if int(scalar.scale_degree) not in (0, 1):
        raise ValueError(
            f"{op_name}: scalar scale_degree must be 0 or 1, got {scalar.scale_degree}"
        )
    if int(scalar.scale_degree) == 1 and int(cipher.state.scale_degree) != 1:
        raise ValueError(
            f"{op_name}: scaled multiplication requires normalized cipher scale_degree=1, "
            f"got {cipher.state.scale_degree}"
        )


class _ScalarMetadataView:
    def __init__(self, scalar):
        self.state = scalar


def _scalar_residues_for_cipher(scalar, cipher):
    residues = scalar.residues
    component = cipher.cv[0]
    device = getattr(component, "device", None)
    if device is not None and hasattr(residues, "to"):
        residues = residues.to(device)
        dtype = getattr(component, "dtype", None)
        if dtype is not None:
            residues = residues.to(dtype=dtype)
    return residues


def homo_add(in0, in1, context):
    validation.validate_binary_cipher_op("homo_add", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, context)
    return _cipher_add(in0, in1, context)


def homo_add_inplace(in0, in1, context):
    _validate_inplace_add_or_sub("homo_add_inplace", in0, in1)
    return _cipher_add_inplace(in0, in1, context)


def homo_sub(in0, in1, context):
    validation.validate_binary_cipher_op("homo_sub", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, context)
    return _cipher_sub(in0, in1, context)


def homo_sub_inplace(in0, in1, context):
    _validate_inplace_add_or_sub("homo_sub_inplace", in0, in1)
    return _cipher_sub_inplace(in0, in1, context)


def homo_mul_relin(in0, in1, context):
    validation.validate_binary_cipher_op("homo_mul_relin", in0, in1, require_ext=False, require_same_metadata=("slots",))
    in0, in1 = _align_for_mul(in0, in1, context)
    product = _cipher_mul(in0, in1, context)
    key_switched = F.cv_keyswitch(
        product.cv[2],
        product.state.cur_limbs,
        context.L,
        context.mult_swk_bx,
        context.mult_swk_ax,
        context,
    )
    return product.cipher_like(
        [
            F.cv_add(product.cv[0], key_switched[0], context.moduliQ, product.state.cur_limbs),
            F.cv_add(product.cv[1], key_switched[1], context.moduliQ, product.state.cur_limbs),
        ]
    )


def homo_mul_no_relin(in0, in1, context):
    validation.validate_binary_cipher_op(
        "homo_mul_no_relin",
        in0,
        in1,
        require_ext=False,
        require_components=2,
        require_same_metadata=("slots", "batch_size"),
    )
    in0, in1 = _align_for_mul(in0, in1, context)
    return _cipher_mul(in0, in1, context)


def homo_add_pt(cipher: Cipher, plaintext: Plaintext, context):
    validation.validate_cipher_plain_op(
        "homo_add_pt",
        cipher,
        plaintext,
        require_same_metadata=("cur_limbs", "scale_degree", "scaling_factor", "slots"),
    )
    return _cipher_add_plain(cipher, plaintext, context)


def homo_add_pt_inplace(cipher: Cipher, plaintext: Plaintext, context):
    validation.validate_cipher_plain_op(
        "homo_add_pt_inplace",
        cipher,
        plaintext,
        require_same_metadata=("cur_limbs", "scale_degree", "scaling_factor", "slots"),
    )
    return _cipher_add_plain_inplace(cipher, plaintext, context)


def homo_mul_pt(cipher: Cipher, plaintext: Plaintext, context):
    if _is_flexible(context):
        _validate_flexible_mul_plain_inputs("homo_mul_pt", cipher, plaintext)
        return _cipher_mul_plain(cipher, plaintext, context)
    validation.validate_cipher_plain_op(
        "homo_mul_pt",
        cipher,
        plaintext,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_mul_plain(cipher, plaintext, context)


def homo_mul_pt_inplace(cipher: Cipher, plaintext: Plaintext, context):
    if _is_flexible(context):
        _validate_flexible_mul_plain_inputs("homo_mul_pt_inplace", cipher, plaintext)
        return _cipher_mul_plain_inplace(cipher, plaintext, context)
    validation.validate_cipher_plain_op(
        "homo_mul_pt_inplace",
        cipher,
        plaintext,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_mul_plain_inplace(cipher, plaintext, context)


def homo_mul_pt_rescale(cipher: Cipher, plaintext: Plaintext, context):
    """Multiply by a plaintext and consume the single u64 rescale limb."""

    return alignment.rescale(homo_mul_pt(cipher, plaintext, context), context)


def homo_add_scalar(cipher, scalar, context):
    scalar = _require_encoded_scalar(scalar, "homo_add_scalar")
    _validate_scalar_add_sub("homo_add_scalar", cipher, scalar)
    return _cipher_add_scalar(cipher, _scalar_residues_for_cipher(scalar, cipher), context)


def homo_add_scalar_inplace(cipher, scalar, context):
    scalar = _require_encoded_scalar(scalar, "homo_add_scalar_inplace")
    _validate_scalar_add_sub("homo_add_scalar_inplace", cipher, scalar)
    return _cipher_add_scalar_inplace(
        cipher,
        _scalar_residues_for_cipher(scalar, cipher),
        context,
    )


def homo_sub_scalar(cipher, scalar, context):
    scalar = _require_encoded_scalar(scalar, "homo_sub_scalar")
    _validate_scalar_add_sub("homo_sub_scalar", cipher, scalar)
    return _cipher_sub_scalar(cipher, _scalar_residues_for_cipher(scalar, cipher), context)


def homo_sub_scalar_inplace(cipher, scalar, context):
    scalar = _require_encoded_scalar(scalar, "homo_sub_scalar_inplace")
    _validate_scalar_add_sub("homo_sub_scalar_inplace", cipher, scalar)
    return _cipher_sub_scalar_inplace(
        cipher,
        _scalar_residues_for_cipher(scalar, cipher),
        context,
    )


def homo_mul_scalar(cipher, scalar, context):
    scalar = _require_encoded_scalar(scalar, "homo_mul_scalar")
    _validate_scalar_mul("homo_mul_scalar", cipher, scalar)
    residues = _scalar_residues_for_cipher(scalar, cipher)
    if int(scalar.scale_degree) == 0:
        return _cipher_mul_scalar_int(cipher, residues, context)
    return _cipher_mul_scalar_double(
        cipher,
        residues,
        context,
        scaling_factor=scalar.scaling_factor,
    )


def homo_mul_scalar_inplace(cipher, scalar, context):
    scalar = _require_encoded_scalar(scalar, "homo_mul_scalar_inplace")
    _validate_scalar_mul("homo_mul_scalar_inplace", cipher, scalar)
    residues = _scalar_residues_for_cipher(scalar, cipher)
    if int(scalar.scale_degree) == 0:
        return _cipher_mul_scalar_int_inplace(cipher, residues, context)
    return _cipher_mul_scalar_double_inplace(
        cipher,
        residues,
        context,
        scaling_factor=scalar.scaling_factor,
    )


def homo_mul_scalar_rescale(cipher, scalar, context):
    """Multiply by a scaled scalar and consume the single u64 rescale limb."""

    scalar = _require_encoded_scalar(scalar, "homo_mul_scalar_rescale")
    if int(scalar.scale_degree) != 1:
        raise ValueError(
            "homo_mul_scalar_rescale requires a scaled scalar with scale_degree=1, "
            f"got {scalar.scale_degree}"
        )
    return alignment.rescale(homo_mul_scalar(cipher, scalar, context), context)


def grouped_pairwise_mac(ciphers, plaintexts, groups, context):
    validation.require_batched_cipher("grouped_pairwise_mac", ciphers, "ciphers")
    validation.require_batched_cipher("grouped_pairwise_mac", plaintexts, "plaintexts")
    groups = validation.validate_positive_int("grouped_pairwise_mac", "groups", groups)

    expected_plaintexts = groups * ciphers.batch_size
    if plaintexts.batch_size != expected_plaintexts:
        raise ValueError(
            "grouped_pairwise_mac: plaintext batch size must equal groups * cipher batch size, "
            f"got {plaintexts.batch_size} != {groups} * {ciphers.batch_size}"
        )

    if _is_flexible(context):
        _validate_flexible_mul_plain_inputs("grouped_pairwise_mac", ciphers, plaintexts)
    else:
        validation.validate_cipher_plain_op(
            "grouped_pairwise_mac",
            ciphers,
            plaintexts,
            require_same_metadata=("cur_limbs", "scale_degree", "scaling_factor", "slots"),
        )
    cv = F.cipher_grouped_pairwise_mac(ciphers, plaintexts, groups, context)
    return ciphers.cipher_like(
        list(cv),
        state=CipherState(
            ciphers.state.cur_limbs,
            ciphers.state.scale_degree + plaintexts.state.scale_degree,
            ciphers.state.scaling_factor * plaintexts.state.scaling_factor,
        ),
        batch_size=groups,
    )


def grouped_pairwise_mac_rescale(ciphers, plaintexts, groups, context):
    """Run grouped plaintext MAC and consume the single u64 rescale limb."""

    return alignment.rescale(
        grouped_pairwise_mac(ciphers, plaintexts, groups, context),
        context,
    )

def grouped_scalar_weighted_acc(ciphers, scalars, context):
    validation.require_batched_cipher("grouped_scalar_weighted_acc", ciphers, "ciphers")
    scalars = _require_encoded_scalar(scalars, "grouped_scalar_weighted_acc")
    _validate_scalar_mul("grouped_scalar_weighted_acc", ciphers, scalars)
    residues = _scalar_residues_for_cipher(scalars, ciphers)
    cv = F.cipher_grouped_scalar_weighted_acc(ciphers, residues, context)
    groups = int(scalars.shape[0])
    return ciphers.cipher_like(
        _grouped_acc_components(cv, groups, ciphers.state.cur_limbs, context.N),
        state=CipherState(
            ciphers.state.cur_limbs,
            ciphers.state.scale_degree + scalars.scale_degree,
            ciphers.state.scaling_factor * scalars.scaling_factor,
        ),
        batch_size=groups,
    )


def _grouped_acc_components(cv, groups, cur_limbs, ring_dim):
    if groups <= 1:
        return list(cv)
    return [
        component.reshape(groups, cur_limbs, ring_dim) if component.dim() == 2 else component
        for component in cv
    ]


def sum_cipher_batch(ciphers, context):
    """Sum every item in a cipher batch into one ciphertext."""

    validation.require_batched_cipher("sum_cipher_batch", ciphers, "ciphers")
    from .layout import unpack_cipher_batch

    items = unpack_cipher_batch(ciphers)
    result = items[0]
    for item in items[1:]:
        result = _cipher_add(result, item, context)
    return result


def homo_mul_relin_rescale_postop(
    in0,
    in1,
    context,
    *,
    apply_double=False,
    add=None,
    sub=None,
    scalar=None,
    plaintext=None,
):
    validation.validate_binary_cipher_op(
        "homo_mul_relin_rescale_postop",
        in0,
        in1,
        require_ext=False,
        require_same_metadata=("slots",),
    )
    post_count = sum(value is not None for value in (add, sub, scalar, plaintext))
    if post_count > 1:
        raise ValueError("homo_mul_relin_rescale_postop accepts at most one post op: add, sub, scalar, or plaintext")

    in0, in1 = _align_for_mul(in0, in1, context)
    if in0.batch_size != 1 or in1.batch_size != 1:
        raise ValueError("homo_mul_relin_rescale_postop only supports batch_size=1 inputs")

    out_cur_limbs = in0.state.cur_limbs - 1
    mod_reduce_factor = context.rescale_divisor_at(out_cur_limbs)
    out_scaling_factor = (
        in0.state.scaling_factor
        * in1.state.scaling_factor
        / mod_reduce_factor
    )
    out_state = CipherState(
        out_cur_limbs,
        in0.state.scale_degree + in1.state.scale_degree - 1,
        out_scaling_factor,
    )

    post_op = _POST_NONE
    post_c0 = post_c1 = post_scalar = None
    if add is not None:
        validation.validate_binary_cipher_op(
            "homo_mul_relin_rescale_postop post add",
            in0,
            add,
            require_ext=False,
            require_same_metadata=("slots",),
        )
        if _is_flexible(context):
            add = alignment.align_to(add, out_state, context)
        else:
            add = alignment.align_to(add, out_state, context)
        post_c0, post_c1 = add.cv
        post_op = _POST_ADD
    elif sub is not None:
        validation.validate_binary_cipher_op(
            "homo_mul_relin_rescale_postop post sub",
            in0,
            sub,
            require_ext=False,
            require_same_metadata=("slots",),
        )
        if _is_flexible(context):
            sub = alignment.align_to(sub, out_state, context)
        else:
            sub = alignment.align_to(sub, out_state, context)
        post_c0, post_c1 = sub.cv
        post_op = _POST_SUB
    elif scalar is not None:
        scalar = _require_encoded_scalar(scalar, "homo_mul_relin_rescale_postop scalar post op")
        if int(scalar.cur_limbs) != int(out_cur_limbs):
            raise ValueError(
                "homo_mul_relin_rescale_postop scalar cur_limbs mismatch: "
                f"{scalar.cur_limbs} != {out_cur_limbs}"
            )
        validation.validate_matching_metadata(
            "homo_mul_relin_rescale_postop scalar post op",
            in0.cipher_like(in0.cv, state=out_state),
            _ScalarMetadataView(scalar),
            ("scale_degree", "scaling_factor"),
        )
        post_scalar = _scalar_residues_for_cipher(scalar, in0)
        post_op = _POST_SCALAR
    elif plaintext is not None:
        validation.validate_cipher_plain_op(
            "homo_mul_relin_rescale_postop post plaintext",
            in0.cipher_like(in0.cv, state=out_state),
            plaintext,
            require_ext=False,
            require_scale_degree=1,
            require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
        )
        post_c0 = plaintext.cv[0]
        if post_c0.dim() == 3:
            if post_c0.size(0) != 1:
                raise ValueError("homo_mul_relin_rescale_postop post plaintext only supports batch_size=1")
            post_c0 = post_c0[0]
        post_op = _POST_PLAINTEXT

    res_c0, res_c1 = F.cv_hmul_relin_rescale(
        in0.cv[0],
        in0.cv[1],
        in1.cv[0],
        in1.cv[1],
        in0.state.cur_limbs,
        context.L,
        context.mult_swk_bx,
        context.mult_swk_ax,
        context,
        apply_double=apply_double,
        post_op=post_op,
        post_c0=post_c0,
        post_c1=post_c1,
        post_scalar=post_scalar,
    )
    return in0.cipher_like([res_c0, res_c1], state=out_state)


def homo_mul_relin_rescale_add_scalar(in0, in1, scalar, context):
    return homo_mul_relin_rescale_postop(
        in0,
        in1,
        context,
        apply_double=False,
        scalar=scalar,
    )


def homo_mul_relin_rescale_add_pt(in0, in1, plaintext, context):
    return homo_mul_relin_rescale_postop(
        in0,
        in1,
        context,
        apply_double=False,
        plaintext=plaintext,
    )
