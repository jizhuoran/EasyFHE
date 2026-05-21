from . import kernels as F


def _scalar_tensor(scalar, cryptoContext, cur_limbs, device):
    if hasattr(scalar, "to") and hasattr(scalar, "dim"):
        return scalar.to(device)
    return F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, cur_limbs).to(device)


def _fused_cuda_available(name, *tensors):
    return (
        F.native_op_available(name)
        and tensors
        and all(hasattr(tensor, "is_cuda") and tensor.is_cuda for tensor in tensors)
    )


def _can_fuse_pairwise(cipher):
    return len(cipher.cv) == 2 and not cipher.is_ext


def _assign_out(out, value):
    return value if out is None else out.replace_with(value)


def _component_shape(component, active_limbs):
    shape = list(component.shape)
    shape[-2] = int(active_limbs)
    return tuple(shape)


def _can_write_out(out, template, active_limbs, cv_count=None):
    if out is None:
        return False
    if cv_count is None:
        cv_count = len(template.cv)
    if len(out.cv) != cv_count:
        return False
    expected = _component_shape(template.cv[0], active_limbs)
    return all(tuple(component.shape) == expected for component in out.cv[:cv_count])


def _require_write_out(out, template, active_limbs, cv_count=None, op_name="cipher op"):
    if out is None:
        return False
    if not _can_write_out(out, template, active_limbs, cv_count=cv_count):
        raise ValueError(f"{op_name}(out=...): out must have writable component tensors")
    return True


def _metadata_like(cipher, cv, **metadata):
    return cipher.cipher_like(cv, **metadata)


def _finish_out(out, template, cv, **metadata):
    value = _metadata_like(template, cv, **metadata)
    return _assign_out(out, value)


def _cipher_add(in0, in1, cryptoContext, *, out=None):
    write_out = _require_write_out(out, in0, in0.cur_limbs, cv_count=2, op_name="homo_add")
    if write_out and _fused_cuda_available(
        "fused_add_mod_write",
        out.cv[0],
        out.cv[1],
        in0.cv[0],
        in0.cv[1],
        in1.cv[0],
        in1.cv[1],
        cryptoContext.moduliQ,
    ):
        F.cv_fused_add_pair_write(
            out.cv[0],
            out.cv[1],
            in0.cv[0],
            in0.cv[1],
            in1.cv[0],
            in1.cv[1],
            cryptoContext.moduliQ,
            in0.cur_limbs,
        )
        return out.replace_with(in0.cipher_like(out.cv))

    if _can_fuse_pairwise(in0) and _can_fuse_pairwise(in1) and _fused_cuda_available(
        "fused_add_mod",
        in0.cv[0],
        in0.cv[1],
        in1.cv[0],
        in1.cv[1],
        cryptoContext.moduliQ,
    ):
        return in0.cipher_like(
            list(
                F.cv_fused_add_pair(
                    in0.cv[0],
                    in0.cv[1],
                    in1.cv[0],
                    in1.cv[1],
                    cryptoContext.moduliQ,
                    in0.cur_limbs,
                )
            )
        )
    if write_out:
        for index, (cv0, cv1) in enumerate(zip(in0.cv, in1.cv)):
            F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs, out=out.cv[index])
        return out.replace_with(in0.cipher_like(out.cv))
    cv = [F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
    return _finish_out(out, in0, cv)


def _cipher_add_ext(in0, in1, cryptoContext, *, out=None):
    active_limbs = in0.cur_limbs + cryptoContext.K
    if _require_write_out(out, in0, active_limbs, op_name="homo_add ext"):
        for index, (cv0, cv1) in enumerate(zip(in0.cv, in1.cv)):
            F.cv_add(
                cv0,
                cv1,
                cryptoContext.QplusP_map[in0.cur_limbs],
                active_limbs,
                out=out.cv[index],
            )
        return out.replace_with(in0.cipher_like(out.cv))
    cv = [
        F.cv_add(
            cv0,
            cv1,
            cryptoContext.QplusP_map[in0.cur_limbs],
            active_limbs,
        )
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return _finish_out(out, in0, cv)


def _cipher_sub(in0, in1, cryptoContext, *, out=None):
    write_out = _require_write_out(out, in0, in0.cur_limbs, cv_count=2, op_name="homo_sub")
    if write_out and _fused_cuda_available(
        "fused_sub_mod_write",
        out.cv[0],
        out.cv[1],
        in0.cv[0],
        in0.cv[1],
        in1.cv[0],
        in1.cv[1],
        cryptoContext.moduliQ,
    ):
        F.cv_fused_sub_pair_write(
            out.cv[0],
            out.cv[1],
            in0.cv[0],
            in0.cv[1],
            in1.cv[0],
            in1.cv[1],
            cryptoContext.moduliQ,
            in0.cur_limbs,
        )
        return out.replace_with(in0.cipher_like(out.cv))

    if _can_fuse_pairwise(in0) and _can_fuse_pairwise(in1) and _fused_cuda_available(
        "fused_sub_mod",
        in0.cv[0],
        in0.cv[1],
        in1.cv[0],
        in1.cv[1],
        cryptoContext.moduliQ,
    ):
        return in0.cipher_like(
            list(
                F.cv_fused_sub_pair(
                    in0.cv[0],
                    in0.cv[1],
                    in1.cv[0],
                    in1.cv[1],
                    cryptoContext.moduliQ,
                    in0.cur_limbs,
                )
            )
        )
    if write_out:
        for index, (cv0, cv1) in enumerate(zip(in0.cv, in1.cv)):
            F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs, out=out.cv[index])
        return out.replace_with(in0.cipher_like(out.cv))
    cv = [F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
    return _finish_out(out, in0, cv)


def _cipher_sub_ext(in0, in1, cryptoContext, *, out=None):
    active_limbs = in0.cur_limbs + cryptoContext.K
    if _require_write_out(out, in0, active_limbs, op_name="homo_sub ext"):
        for index, (cv0, cv1) in enumerate(zip(in0.cv, in1.cv)):
            F.cv_sub(
                cv0,
                cv1,
                cryptoContext.QplusP_map[in0.cur_limbs],
                active_limbs,
                out=out.cv[index],
            )
        return out.replace_with(in0.cipher_like(out.cv))
    cv = [
        F.cv_sub(
            cv0,
            cv1,
            cryptoContext.QplusP_map[in0.cur_limbs],
            active_limbs,
        )
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return _finish_out(out, in0, cv)


def _cipher_add_plain(cipher, plaintext, cryptoContext, *, out=None):
    moduli = cryptoContext.QplusP_map[cipher.cur_limbs]
    active_limbs = cipher.cur_limbs + (cryptoContext.K if cipher.is_ext else 0)
    if _require_write_out(out, cipher, active_limbs, cv_count=2, op_name="homo_add_pt"):
        F.cv_add(cipher.cv[0], plaintext.cv[0], moduli, active_limbs, out=out.cv[0])
        if out.cv[1] is not cipher.cv[1]:
            out.cv[1].copy_(cipher.cv[1])
        return out.replace_with(cipher.cipher_like(out.cv))
    cv = [
        F.cv_add(cipher.cv[0], plaintext.cv[0], moduli, active_limbs),
        cipher.cv[1],
    ]
    return _finish_out(out, cipher, cv)


def _cipher_mul_plain(cipher, plaintext, cryptoContext, *, out=None):
    active_limbs = cipher.cur_limbs + (cryptoContext.K if cipher.is_ext else 0)
    write_out = _require_write_out(out, cipher, active_limbs, cv_count=2, op_name="homo_mul_pt")
    if write_out and not cipher.is_ext and _fused_cuda_available(
        "fused_mul_pt_mod_write",
        out.cv[0],
        out.cv[1],
        cipher.cv[0],
        cipher.cv[1],
        plaintext.cv[0],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
    ):
        F.cv_fused_mul_pt_pair_write(
            out.cv[0],
            out.cv[1],
            cipher.cv[0],
            cipher.cv[1],
            plaintext.cv[0],
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            cipher.cur_limbs,
        )
        return out.replace_with(
            cipher.cipher_like(
                out.cv,
                scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
                noise_deg=cipher.noise_deg + plaintext.noise_deg,
            )
        )

    if _can_fuse_pairwise(cipher) and _fused_cuda_available(
        "fused_mul_pt_mod",
        cipher.cv[0],
        cipher.cv[1],
        plaintext.cv[0],
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
    ):
        cv = list(
            F.cv_fused_mul_pt_pair(
                cipher.cv[0],
                cipher.cv[1],
                plaintext.cv[0],
                cryptoContext.moduliQ,
                cryptoContext.q_mu,
                cipher.cur_limbs,
            )
        )
        return cipher.cipher_like(
            cv,
            scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
            noise_deg=cipher.noise_deg + plaintext.noise_deg,
        )

    moduli = cryptoContext.QplusP_map[cipher.cur_limbs]
    mu = cryptoContext.QmuplusPmu_map[cipher.cur_limbs]
    plaintext_values = plaintext.cv[0]
    if write_out:
        F.cv_mul(cipher.cv[0], plaintext_values, moduli, mu, active_limbs, out=out.cv[0])
        F.cv_mul(cipher.cv[1], plaintext_values, moduli, mu, active_limbs, out=out.cv[1])
        return out.replace_with(
            cipher.cipher_like(
                out.cv,
                scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
                noise_deg=cipher.noise_deg + plaintext.noise_deg,
            )
        )
    cv0 = F.cv_mul(cipher.cv[0], plaintext_values, moduli, mu, active_limbs)
    cv1 = F.cv_mul(cipher.cv[1], plaintext_values, moduli, mu, active_limbs)
    return _assign_out(out, cipher.cipher_like(
        [cv0, cv1],
        scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
        noise_deg=cipher.noise_deg + plaintext.noise_deg,
    ))


def _cipher_mul(in0, in1, cryptoContext):
    bx = F.cv_mul(in0.cv[0], in1.cv[0], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
    ax = F.cv_add(
        F.cv_mul(in0.cv[0], in1.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs),
        F.cv_mul(in0.cv[1], in1.cv[0], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs),
        cryptoContext.moduliQ,
        in0.cur_limbs,
    )
    axax = F.cv_mul(in0.cv[1], in1.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
    sc_factor = cryptoContext.scale_at(in0.cur_limbs)
    return in0.cipher_like(
        [bx, ax, axax],
        scaling_factor=in0.scaling_factor * sc_factor,
        noise_deg=in0.noise_deg + in1.noise_deg,
    )


def _cipher_square(in0, cryptoContext):
    bx = F.cv_mul(in0.cv[0], in0.cv[0], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
    ax = F.cv_mul(in0.cv[0], in0.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
    ax = F.cv_add(ax, ax, cryptoContext.moduliQ, in0.cur_limbs)
    axax = F.cv_mul(in0.cv[1], in0.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
    sc_factor = cryptoContext.scale_at(in0.cur_limbs)
    return in0.cipher_like(
        [bx, ax, axax],
        scaling_factor=in0.scaling_factor * sc_factor,
        noise_deg=in0.noise_deg + in0.noise_deg,
    )


def _cipher_add_scalar(in0, scalar, cryptoContext, *, out=None):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.cur_limbs, in0.cv[0].device)
    if _require_write_out(out, in0, in0.cur_limbs, cv_count=2, op_name="homo_add_scalar"):
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs, out=out.cv[0])
        if out.cv[1] is not in0.cv[1]:
            out.cv[1].copy_(in0.cv[1])
        return out.replace_with(in0.cipher_like(out.cv))
    return _finish_out(out, in0, [
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ])


def _cipher_sub_scalar(in0, scalar, cryptoContext, *, out=None):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.cur_limbs, in0.cv[0].device)
    if _require_write_out(out, in0, in0.cur_limbs, cv_count=2, op_name="homo_sub_scalar"):
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs, out=out.cv[0])
        if out.cv[1] is not in0.cv[1]:
            out.cv[1].copy_(in0.cv[1])
        return out.replace_with(in0.cipher_like(out.cv))
    return _finish_out(out, in0, [
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ])


def _cipher_mul_scalar_double(in0, scalar, cryptoContext, *, out=None):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.cur_limbs, in0.cv[0].device)
    write_out = _require_write_out(out, in0, in0.cur_limbs, cv_count=2, op_name="homo_mul_scalar_double")
    if write_out and _fused_cuda_available(
        "fused_mul_scalar_mod_write",
        out.cv[0],
        out.cv[1],
        in0.cv[0],
        in0.cv[1],
        scalar_mod,
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
    ):
        sc_factor = cryptoContext.scale_at(in0.cur_limbs)
        F.cv_fused_mul_scalar_pair_write(
            out.cv[0],
            out.cv[1],
            in0.cv[0],
            in0.cv[1],
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        )
        return out.replace_with(
            in0.cipher_like(
                out.cv,
                scaling_factor=in0.scaling_factor * sc_factor,
                noise_deg=in0.noise_deg + 1,
            )
        )
    if _can_fuse_pairwise(in0) and _fused_cuda_available(
        "fused_mul_scalar_mod",
        in0.cv[0],
        in0.cv[1],
        scalar_mod,
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
    ):
        sc_factor = cryptoContext.scale_at(in0.cur_limbs)
        cv = list(
            F.cv_fused_mul_scalar_pair(
                in0.cv[0],
                in0.cv[1],
                scalar_mod,
                cryptoContext.moduliQ,
                cryptoContext.q_mu,
                in0.cur_limbs,
            )
        )
        return in0.cipher_like(
            cv,
            scaling_factor=in0.scaling_factor * sc_factor,
            noise_deg=in0.noise_deg + 1,
        )
    sc_factor = cryptoContext.scale_at(in0.cur_limbs)
    if write_out:
        for index, cv0 in enumerate(in0.cv):
            F.cv_mul_scalar(
                cv0,
                scalar_mod,
                cryptoContext.moduliQ,
                cryptoContext.q_mu,
                in0.cur_limbs,
                out=out.cv[index],
            )
        return out.replace_with(
            in0.cipher_like(
                out.cv,
                scaling_factor=in0.scaling_factor * sc_factor,
                noise_deg=in0.noise_deg + 1,
            )
        )
    cv = [
        F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
        for cv0 in in0.cv
    ]
    return _finish_out(out, in0, cv, scaling_factor=in0.scaling_factor * sc_factor, noise_deg=in0.noise_deg + 1)


def _cipher_mul_scalar_int(in0, scalar, cryptoContext, *, out=None):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.cur_limbs, in0.cv[0].device)
    write_out = _require_write_out(out, in0, in0.cur_limbs, cv_count=2, op_name="homo_mul_scalar_int")
    if write_out and _fused_cuda_available(
        "fused_mul_scalar_mod_write",
        out.cv[0],
        out.cv[1],
        in0.cv[0],
        in0.cv[1],
        scalar_mod,
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
    ):
        F.cv_fused_mul_scalar_pair_write(
            out.cv[0],
            out.cv[1],
            in0.cv[0],
            in0.cv[1],
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.cur_limbs,
        )
        return out.replace_with(
            in0.cipher_like(out.cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)
        )
    if _can_fuse_pairwise(in0) and _fused_cuda_available(
        "fused_mul_scalar_mod",
        in0.cv[0],
        in0.cv[1],
        scalar_mod,
        cryptoContext.moduliQ,
        cryptoContext.q_mu,
    ):
        cv = list(
            F.cv_fused_mul_scalar_pair(
                in0.cv[0],
                in0.cv[1],
                scalar_mod,
                cryptoContext.moduliQ,
                cryptoContext.q_mu,
                in0.cur_limbs,
            )
        )
        return in0.cipher_like(
            cv,
            scaling_factor=in0.scaling_factor,
            noise_deg=in0.noise_deg,
        )
    if write_out:
        for index, cv0 in enumerate(in0.cv):
            F.cv_mul_scalar(
                cv0,
                scalar_mod,
                cryptoContext.moduliQ,
                cryptoContext.q_mu,
                in0.cur_limbs,
                out=out.cv[index],
            )
        return out.replace_with(
            in0.cipher_like(out.cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)
        )
    cv = [
        F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
        for cv0 in in0.cv
    ]
    return _finish_out(out, in0, cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)


def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)
