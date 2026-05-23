from . import kernels as F
from ..ciphertext import Cipher, CipherState, Plaintext
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


def _align_for_add_or_sub(in0, in1, cryptoContext):
    target = alignment.plan_add_alignment(in0, in1, cryptoContext)
    return alignment.align_to(in0, target, cryptoContext), alignment.align_to(in1, target, cryptoContext)


def _align_for_mul(ct1: Cipher, ct2: Cipher, cryptoContext):
    target1, target2 = alignment.plan_mul_alignment(ct1, ct2, cryptoContext)
    return alignment.align_to(ct1, target1, cryptoContext), alignment.align_to(ct2, target2, cryptoContext)


def _validate_inplace_add_or_sub(op_name, in0, in1):
    validation.validate_binary_cipher_op(
        op_name,
        in0,
        in1,
        require_same_metadata=("slots", "cur_limbs", "noise_deg", "scaling_factor"),
    )


def _reduce_scalar_to_crt(value, cur_limbs, moduli):
    return [int(value) % int(moduli[i]) for i in range(cur_limbs)]


def _encode_double_for_scalar_op(constant, cur_limbs, cryptoContext):
    scale = cryptoContext.scale_at(cur_limbs)
    encoded = int(constant * scale + 0.5)
    return _reduce_scalar_to_crt(encoded, cur_limbs, cryptoContext.moduliQ_scalar)


def _encode_int_for_scalar_op(value, cur_limbs, cryptoContext):
    return _reduce_scalar_to_crt(int(value), cur_limbs, cryptoContext.moduliQ_scalar)


def _require_encoded_scalar(value, op_name):
    return validation.require_encoded_scalar(value, op_name)


def homo_add(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_add", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_add(in0, in1, cryptoContext)


def homo_add_inplace(in0, in1, cryptoContext):
    _validate_inplace_add_or_sub("homo_add_inplace", in0, in1)
    return _cipher_add_inplace(in0, in1, cryptoContext)


def homo_sub(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_sub", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    return _cipher_sub(in0, in1, cryptoContext)


def homo_sub_inplace(in0, in1, cryptoContext):
    _validate_inplace_add_or_sub("homo_sub_inplace", in0, in1)
    return _cipher_sub_inplace(in0, in1, cryptoContext)


def homo_mul_relin(in0, in1, cryptoContext):
    validation.validate_binary_cipher_op("homo_mul_relin", in0, in1, require_ext=False, require_same_metadata=("slots",))
    in0, in1 = _align_for_mul(in0, in1, cryptoContext)
    product = _cipher_mul(in0, in1, cryptoContext)
    key_switched = F.cv_keyswitch(
        product.cv[2],
        product.state.cur_limbs,
        cryptoContext.L,
        cryptoContext.mult_swk_bx,
        cryptoContext.mult_swk_ax,
        cryptoContext,
    )
    return product.cipher_like(
        [
            F.cv_add(product.cv[0], key_switched[0], cryptoContext.moduliQ, product.state.cur_limbs),
            F.cv_add(product.cv[1], key_switched[1], cryptoContext.moduliQ, product.state.cur_limbs),
        ]
    )


def homo_add_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    validation.validate_cipher_plain_op(
        "homo_add_pt",
        cipher,
        plaintext,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_add_plain(cipher, plaintext, cryptoContext)


def homo_add_pt_inplace(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    validation.validate_cipher_plain_op(
        "homo_add_pt_inplace",
        cipher,
        plaintext,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_add_plain_inplace(cipher, plaintext, cryptoContext)


def homo_mul_pt(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    validation.validate_cipher_plain_op(
        "homo_mul_pt",
        cipher,
        plaintext,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_mul_plain(cipher, plaintext, cryptoContext)


def homo_mul_pt_inplace(cipher: Cipher, plaintext: Plaintext, cryptoContext):
    validation.validate_cipher_plain_op(
        "homo_mul_pt_inplace",
        cipher,
        plaintext,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    return _cipher_mul_plain_inplace(cipher, plaintext, cryptoContext)


def homo_add_scalar_double(cipher, constant, cryptoContext):
    validation.validate_cipher_op(
        "homo_add_scalar_double",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_add_scalar(cipher, _require_encoded_scalar(constant, "homo_add_scalar_double"), cryptoContext)


def homo_add_scalar_double_inplace(cipher, constant, cryptoContext):
    validation.validate_cipher_op(
        "homo_add_scalar_double_inplace",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_add_scalar_inplace(
        cipher,
        _require_encoded_scalar(constant, "homo_add_scalar_double_inplace"),
        cryptoContext,
    )


def homo_add_scalar_int(cipher, scalar, cryptoContext):
    validation.validate_cipher_op(
        "homo_add_scalar_int",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_add_scalar(cipher, _require_encoded_scalar(scalar, "homo_add_scalar_int"), cryptoContext)


def homo_add_scalar_int_inplace(cipher, scalar, cryptoContext):
    validation.validate_cipher_op(
        "homo_add_scalar_int_inplace",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_add_scalar_inplace(
        cipher,
        _require_encoded_scalar(scalar, "homo_add_scalar_int_inplace"),
        cryptoContext,
    )


def homo_sub_scalar_int(cipher, scalar, cryptoContext):
    validation.validate_cipher_op(
        "homo_sub_scalar_int",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_sub_scalar(cipher, _require_encoded_scalar(scalar, "homo_sub_scalar_int"), cryptoContext)


def homo_sub_scalar_int_inplace(cipher, scalar, cryptoContext):
    validation.validate_cipher_op(
        "homo_sub_scalar_int_inplace",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_sub_scalar_inplace(
        cipher,
        _require_encoded_scalar(scalar, "homo_sub_scalar_int_inplace"),
        cryptoContext,
    )


def homo_mul_scalar_int(cipher, scalar, cryptoContext):
    validation.validate_cipher_op(
        "homo_mul_scalar_int",
        cipher,
        require_ext=False,
    )
    return _cipher_mul_scalar_int(cipher, _require_encoded_scalar(scalar, "homo_mul_scalar_int"), cryptoContext)


def homo_mul_scalar_int_inplace(cipher, scalar, cryptoContext):
    validation.validate_cipher_op(
        "homo_mul_scalar_int_inplace",
        cipher,
        require_ext=False,
    )
    return _cipher_mul_scalar_int_inplace(
        cipher,
        _require_encoded_scalar(scalar, "homo_mul_scalar_int_inplace"),
        cryptoContext,
    )


def homo_mul_scalar_double(cipher, constant, cryptoContext):
    validation.validate_cipher_op(
        "homo_mul_scalar_double",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_mul_scalar_double(
        cipher,
        _require_encoded_scalar(constant, "homo_mul_scalar_double"),
        cryptoContext,
    )


def homo_mul_scalar_double_inplace(cipher, constant, cryptoContext):
    validation.validate_cipher_op(
        "homo_mul_scalar_double_inplace",
        cipher,
        require_ext=False,
        require_noise_deg=1,
    )
    return _cipher_mul_scalar_double_inplace(
        cipher,
        _require_encoded_scalar(constant, "homo_mul_scalar_double_inplace"),
        cryptoContext,
    )


def grouped_pairwise_mac(ciphers, plaintexts, groups, cryptoContext):
    validation.require_batched_cipher("grouped_pairwise_mac", ciphers, "ciphers")
    validation.require_batched_cipher("grouped_pairwise_mac", plaintexts, "plaintexts")
    groups = validation.validate_positive_int("grouped_pairwise_mac", "groups", groups)

    expected_plaintexts = groups * ciphers.batch_size
    if plaintexts.batch_size != expected_plaintexts:
        raise ValueError(
            "grouped_pairwise_mac: plaintext batch size must equal groups * cipher batch size, "
            f"got {plaintexts.batch_size} != {groups} * {ciphers.batch_size}"
        )

    validation.validate_cipher_plain_op(
        "grouped_pairwise_mac",
        ciphers,
        plaintexts,
        require_noise_deg=1,
        require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
    )
    cv = F.cipher_grouped_pairwise_mac(ciphers, plaintexts, groups, cryptoContext)
    return ciphers.cipher_like(
        list(cv),
        state=CipherState(
            ciphers.state.cur_limbs,
            ciphers.state.noise_deg + plaintexts.state.noise_deg,
            ciphers.state.scaling_factor * plaintexts.state.scaling_factor,
        ),
        batch_size=groups,
    )


def grouped_scalar_weighted_acc(ciphers, scalars, cryptoContext):
    validation.require_batched_cipher("grouped_scalar_weighted_acc", ciphers, "ciphers")
    cv = F.cipher_grouped_scalar_weighted_acc(ciphers, scalars, cryptoContext)
    return ciphers.cipher_like(
        list(cv),
        state=CipherState(
            ciphers.state.cur_limbs,
            ciphers.state.noise_deg + 1,
            ciphers.state.scaling_factor * cryptoContext.scale_at(ciphers.state.cur_limbs),
        ),
        batch_size=int(scalars.shape[0]),
    )


def homo_mul_relin_rescale_postop(
    in0,
    in1,
    cryptoContext,
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

    in0, in1 = _align_for_mul(in0, in1, cryptoContext)
    if in0.batch_size != 1 or in1.batch_size != 1:
        raise ValueError("homo_mul_relin_rescale_postop only supports batch_size=1 inputs")

    out_cur_limbs = in0.state.cur_limbs - 1
    mod_reduce_factor = cryptoContext.rescale_divisor_at(out_cur_limbs)
    out_scaling_factor = (
        in0.state.scaling_factor
        * cryptoContext.scale_at(in0.state.cur_limbs)
        / mod_reduce_factor
    )
    out_state = CipherState(
        out_cur_limbs,
        in0.state.noise_deg + in1.state.noise_deg - 1,
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
        add = alignment.align_to(add, out_state, cryptoContext)
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
        sub = alignment.align_to(sub, out_state, cryptoContext)
        post_c0, post_c1 = sub.cv
        post_op = _POST_SUB
    elif scalar is not None:
        post_scalar = F.gen_scalar_tensor(
            _require_encoded_scalar(scalar, "homo_mul_relin_rescale_postop scalar post op"),
            cryptoContext.moduliQ_scalar,
            out_cur_limbs,
        ).to(in0.cv[0].device)
        post_op = _POST_SCALAR
    elif plaintext is not None:
        validation.validate_cipher_plain_op(
            "homo_mul_relin_rescale_postop post plaintext",
            in0.cipher_like(in0.cv, state=out_state),
            plaintext,
            require_ext=False,
            require_noise_deg=1,
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
        cryptoContext.L,
        cryptoContext.mult_swk_bx,
        cryptoContext.mult_swk_ax,
        cryptoContext,
        apply_double=apply_double,
        post_op=post_op,
        post_c0=post_c0,
        post_c1=post_c1,
        post_scalar=post_scalar,
    )
    return in0.cipher_like([res_c0, res_c1], state=out_state)


def homo_mul_relin_rescale_add_scalar(in0, in1, scalar, cryptoContext):
    return homo_mul_relin_rescale_postop(
        in0,
        in1,
        cryptoContext,
        apply_double=False,
        scalar=scalar,
    )


def homo_mul_relin_rescale_add_pt(in0, in1, plaintext, cryptoContext):
    return homo_mul_relin_rescale_postop(
        in0,
        in1,
        cryptoContext,
        apply_double=False,
        plaintext=plaintext,
    )
