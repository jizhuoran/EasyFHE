from . import kernels as F
from ..ciphertext import Cipher
from ..runtime import validation
from . import alignment
from .plaintext import _encode_double_for_scalar_op, homo_add_pt, homo_add_scalar_double
from .primitives import _cipher_add, _cipher_add_ext, _cipher_mul, _cipher_square, _cipher_sub, _cipher_sub_ext


def _assign_out(out, value):
    return value if out is None else out.replace_with(value)


def _align_for_add_or_sub(in0, in1, cryptoContext):
    target = alignment.plan_add_alignment(in0, in1, cryptoContext)
    return alignment.align_to(in0, target, cryptoContext), alignment.align_to(in1, target, cryptoContext)


def _align_for_mul(ct1: Cipher, ct2: Cipher, cryptoContext):
    target1, target2 = alignment.plan_mul_alignment(ct1, ct2, cryptoContext)
    return alignment.align_to(ct1, target1, cryptoContext), alignment.align_to(ct2, target2, cryptoContext)


def homo_add(in0, in1, cryptoContext, *, out=None):
    validation.validate_binary_cipher_op("homo_add", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_add_ext(in0, in1, cryptoContext, out=out)
    return _cipher_add(in0, in1, cryptoContext, out=out)


def homo_add_inplace(in0, in1, cryptoContext):
    return homo_add(in0, in1, cryptoContext, out=in0)


def homo_sub(in0, in1, cryptoContext, *, out=None):
    validation.validate_binary_cipher_op("homo_sub", in0, in1, require_same_metadata=("slots",))
    in0, in1 = _align_for_add_or_sub(in0, in1, cryptoContext)
    if in0.is_ext:
        return _cipher_sub_ext(in0, in1, cryptoContext, out=out)
    return _cipher_sub(in0, in1, cryptoContext, out=out)


def homo_sub_inplace(in0, in1, cryptoContext):
    return homo_sub(in0, in1, cryptoContext, out=in0)


def homo_mul(in0, in1, cryptoContext, *, out=None):
    validation.validate_binary_cipher_op("homo_mul", in0, in1, require_ext=False, require_same_metadata=("slots",))
    in0, in1 = _align_for_mul(in0, in1, cryptoContext)
    return _assign_out(out, _relinearize(_cipher_mul(in0, in1, cryptoContext), cryptoContext))


def _post_mul_rescale_fallback(value, cryptoContext, *, add=None, sub=None, scalar=None, plaintext=None):
    if add is not None:
        value = homo_add(value, add, cryptoContext)
    elif sub is not None:
        value = homo_sub(value, sub, cryptoContext)
    elif scalar is not None:
        value = homo_add_scalar_double(value, scalar, cryptoContext)
    elif plaintext is not None:
        value = homo_add_pt(value, plaintext, cryptoContext)
    return value


def homo_mul_rescale(
    in0,
    in1,
    cryptoContext,
    *,
    apply_double=False,
    add=None,
    sub=None,
    scalar=None,
    plaintext=None,
    out=None,
):
    validation.validate_binary_cipher_op(
        "homo_mul_rescale",
        in0,
        in1,
        require_ext=False,
        require_same_metadata=("slots",),
    )
    post_count = sum(value is not None for value in (add, sub, scalar, plaintext))
    if post_count > 1:
        raise ValueError("homo_mul_rescale accepts at most one post op: add, sub, scalar, or plaintext")

    if cryptoContext.scale_mode != "fixed":
        raise ValueError(f"homo_mul_rescale supports only fixed scale mode, got {cryptoContext.scale_mode!r}")

    in0, in1 = _align_for_mul(in0, in1, cryptoContext)
    if in0.batch_size != 1 or in1.batch_size != 1:
        prod = homo_mul(in0, in1, cryptoContext)
        if apply_double:
            prod = _cipher_add(prod, prod, cryptoContext)
        prod = alignment.rescale_one_level(prod, cryptoContext)
        return _assign_out(out, _post_mul_rescale_fallback(
            prod,
            cryptoContext,
            add=add,
            sub=sub,
            scalar=scalar,
            plaintext=plaintext,
        ))

    out_cur_limbs = in0.cur_limbs - 1
    mod_reduce_factor = cryptoContext.rescale_divisor_at(out_cur_limbs)
    out_scaling_factor = (
        in0.scaling_factor
        * cryptoContext.scale_at(in0.cur_limbs)
        / mod_reduce_factor
    )
    out_state = alignment.CipherState(out_cur_limbs, 1, out_scaling_factor)
    post_op = 0
    post_c0 = post_c1 = post_scalar = None
    if add is not None:
        validation.validate_binary_cipher_op(
            "homo_mul_rescale post add",
            in0,
            add,
            require_ext=False,
            require_same_metadata=("slots",),
        )
        add = alignment.align_to(add, out_state, cryptoContext)
        post_c0, post_c1 = add.cv
        post_op = 1
    elif sub is not None:
        validation.validate_binary_cipher_op(
            "homo_mul_rescale post sub",
            in0,
            sub,
            require_ext=False,
            require_same_metadata=("slots",),
        )
        sub = alignment.align_to(sub, out_state, cryptoContext)
        post_c0, post_c1 = sub.cv
        post_op = 2
    elif scalar is not None:
        encoded_abs = _encode_double_for_scalar_op(abs(scalar), out_cur_limbs, cryptoContext)
        if scalar < 0:
            encoded_abs = [-value for value in encoded_abs]
        post_scalar = F.gen_scalar_tensor(encoded_abs, cryptoContext.moduliQ_scalar, out_cur_limbs).to(in0.cv[0].device)
        post_op = 3
    elif plaintext is not None:
        validation.validate_cipher_plain_op(
            "homo_mul_rescale post plaintext",
            in0.cipher_like(
                in0.cv,
                cur_limbs=out_cur_limbs,
                scaling_factor=out_scaling_factor,
                noise_deg=1,
            ),
            plaintext,
            require_ext=False,
            require_noise_deg=1,
            require_same_metadata=("cur_limbs", "scaling_factor", "slots"),
        )
        post_c0 = plaintext.cv[0]
        if post_c0.dim() == 3:
            if post_c0.size(0) != 1:
                raise ValueError("homo_mul_rescale post plaintext only supports batch_size=1")
            post_c0 = post_c0[0]
        post_op = 4

    res = F.cv_hmul_double_rescale(
        in0.cv[0],
        in0.cv[1],
        in1.cv[0],
        in1.cv[1],
        in0.cur_limbs,
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
    return _assign_out(out, in0.cipher_like(
        [res[0, 0], res[1, 0]],
        cur_limbs=out_cur_limbs,
        scaling_factor=out_scaling_factor,
        noise_deg=in0.noise_deg + in1.noise_deg - 1,
    ))


def homo_mul_double_rescale(in0, in1, cryptoContext, *, out=None):
    return homo_mul_rescale(in0, in1, cryptoContext, apply_double=True, out=out)


def homo_mul_rescale_addscalar(in0, in1, scalar, cryptoContext, *, out=None):
    return homo_mul_rescale(
        in0,
        in1,
        cryptoContext,
        apply_double=False,
        scalar=scalar,
        out=out,
    )


def homo_mul_rescale_addpt(in0, in1, plaintext, cryptoContext, *, out=None):
    return homo_mul_rescale(
        in0,
        in1,
        cryptoContext,
        apply_double=False,
        plaintext=plaintext,
        out=out,
    )


def homo_square(in0, cryptoContext, *, out=None):
    validation.validate_cipher_op("homo_square", in0, require_ext=False)
    in0 = alignment.align_to(in0, alignment.plan_reduce_noise_to_one(in0, cryptoContext), cryptoContext)
    return _assign_out(out, _relinearize(_cipher_square(in0, cryptoContext), cryptoContext))


def _relinearize(cipher, cryptoContext):
    key_switched = F.cv_keyswitch(
        cipher.cv[2],
        cipher.cur_limbs,
        cryptoContext.L,
        cryptoContext.mult_swk_bx,
        cryptoContext.mult_swk_ax,
        cryptoContext,
    )
    cv = [
        F.cv_add(cipher.cv[0], key_switched[0], cryptoContext.moduliQ, cipher.cur_limbs),
        F.cv_add(cipher.cv[1], key_switched[1], cryptoContext.moduliQ, cipher.cur_limbs),
    ]
    return cipher.cipher_like(cv)
