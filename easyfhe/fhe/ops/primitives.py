from ..ciphertext import CipherState
from . import kernels as F


def _scalar_tensor(scalar, cryptoContext, cur_limbs, device):
    if hasattr(scalar, "to") and hasattr(scalar, "dim"):
        return scalar.to(device)
    return F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, cur_limbs).to(device)


def _all_cuda(*tensors):
    return tensors and all(hasattr(tensor, "is_cuda") and tensor.is_cuda for tensor in tensors)


def _can_fuse_pairwise(cipher):
    return len(cipher.cv) == 2


def _can_use_pair_kernel(cipher, *extra_tensors):
    return _can_fuse_pairwise(cipher) and _all_cuda(*cipher.cv, *extra_tensors)


def _can_use_binary_pair_kernel(left, right, *extra_tensors):
    return _can_fuse_pairwise(left) and _can_use_pair_kernel(right, *left.cv, *extra_tensors)


def _active_limbs(cipher, cryptoContext):
    return cipher.state.cur_limbs + (cryptoContext.K if cipher.is_ext else 0)


def _moduli_for(cipher, cryptoContext):
    maps = getattr(cryptoContext, "QplusP_map", None)
    if maps is not None:
        return maps[cipher.state.cur_limbs]
    return cryptoContext.moduliQ


def _mu_for(cipher, cryptoContext):
    maps = getattr(cryptoContext, "QmuplusPmu_map", None)
    if maps is not None:
        return maps[cipher.state.cur_limbs]
    return cryptoContext.q_mu


def _cipher_add(in0, in1, cryptoContext):
    active_limbs = _active_limbs(in0, cryptoContext)
    moduli = _moduli_for(in0, cryptoContext)
    if _can_use_binary_pair_kernel(in0, in1, moduli):
        cv = F.cv_add_pair(
            in0.cv[0],
            in0.cv[1],
            in1.cv[0],
            in1.cv[1],
            moduli,
            active_limbs,
        )
        return in0.cipher_like(cv)

    cv = [F.cv_add(cv0, cv1, moduli, active_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
    return in0.cipher_like(cv)


def _cipher_add_inplace(in0, in1, cryptoContext):
    active_limbs = _active_limbs(in0, cryptoContext)
    moduli = _moduli_for(in0, cryptoContext)
    if _can_use_binary_pair_kernel(in0, in1, moduli):
        F.cv_add_pair_(
            in0.cv[0],
            in0.cv[1],
            in1.cv[0],
            in1.cv[1],
            moduli,
            active_limbs,
        )
    else:
        for cv0, cv1 in zip(in0.cv, in1.cv):
            F.cv_add(cv0, cv1, moduli, active_limbs, inplace=True)
    return in0.replace_with(in0.cipher_like(in0.cv))


def _cipher_sub(in0, in1, cryptoContext):
    active_limbs = _active_limbs(in0, cryptoContext)
    moduli = _moduli_for(in0, cryptoContext)
    if _can_use_binary_pair_kernel(in0, in1, moduli):
        cv = F.cv_sub_pair(
            in0.cv[0],
            in0.cv[1],
            in1.cv[0],
            in1.cv[1],
            moduli,
            active_limbs,
        )
        return in0.cipher_like(cv)

    cv = [F.cv_sub(cv0, cv1, moduli, active_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
    return in0.cipher_like(cv)


def _cipher_sub_inplace(in0, in1, cryptoContext):
    active_limbs = _active_limbs(in0, cryptoContext)
    moduli = _moduli_for(in0, cryptoContext)
    if _can_use_binary_pair_kernel(in0, in1, moduli):
        F.cv_sub_pair_(
            in0.cv[0],
            in0.cv[1],
            in1.cv[0],
            in1.cv[1],
            moduli,
            active_limbs,
        )
    else:
        for cv0, cv1 in zip(in0.cv, in1.cv):
            F.cv_sub(cv0, cv1, moduli, active_limbs, inplace=True)
    return in0.replace_with(in0.cipher_like(in0.cv))


def _cipher_add_plain(cipher, plaintext, cryptoContext):
    moduli = _moduli_for(cipher, cryptoContext)
    active_limbs = _active_limbs(cipher, cryptoContext)
    cv = [
        F.cv_add(cipher.cv[0], plaintext.cv[0], moduli, active_limbs),
        cipher.cv[1],
    ]
    return cipher.cipher_like(cv)


def _cipher_add_plain_inplace(cipher, plaintext, cryptoContext):
    moduli = _moduli_for(cipher, cryptoContext)
    active_limbs = _active_limbs(cipher, cryptoContext)
    F.cv_add(cipher.cv[0], plaintext.cv[0], moduli, active_limbs, inplace=True)
    return cipher.replace_with(cipher.cipher_like(cipher.cv))


def _cipher_mul_plain(cipher, plaintext, cryptoContext):
    active_limbs = _active_limbs(cipher, cryptoContext)
    moduli = _moduli_for(cipher, cryptoContext)
    mu = _mu_for(cipher, cryptoContext)
    if _can_use_pair_kernel(cipher, plaintext.cv[0], moduli, mu):
        cv = F.cv_mul_pt_pair(
            cipher.cv[0],
            cipher.cv[1],
            plaintext.cv[0],
            moduli,
            mu,
            active_limbs,
        )
    else:
        cv = [F.cv_mul(component, plaintext.cv[0], moduli, mu, active_limbs) for component in cipher.cv]
    return cipher.cipher_like(
        cv,
        state=CipherState(
            cipher.state.cur_limbs,
            cipher.state.noise_deg + plaintext.state.noise_deg,
            cipher.state.scaling_factor * plaintext.state.scaling_factor,
        ),
    )


def _cipher_mul_plain_inplace(cipher, plaintext, cryptoContext):
    active_limbs = _active_limbs(cipher, cryptoContext)
    moduli = _moduli_for(cipher, cryptoContext)
    mu = _mu_for(cipher, cryptoContext)
    if _can_use_pair_kernel(cipher, plaintext.cv[0], moduli, mu):
        F.cv_mul_pt_pair_(
            cipher.cv[0],
            cipher.cv[1],
            plaintext.cv[0],
            moduli,
            mu,
            active_limbs,
        )
    else:
        for component in cipher.cv:
            F.cv_mul(component, plaintext.cv[0], moduli, mu, active_limbs, inplace=True)
    return cipher.replace_with(
        cipher.cipher_like(
            cipher.cv,
            state=CipherState(
                cipher.state.cur_limbs,
                cipher.state.noise_deg + plaintext.state.noise_deg,
                cipher.state.scaling_factor * plaintext.state.scaling_factor,
            ),
        )
    )


def _cipher_mul(in0, in1, cryptoContext):
    bx = F.cv_mul(in0.cv[0], in1.cv[0], cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs)
    ax = F.cv_add(
        F.cv_mul(in0.cv[0], in1.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs),
        F.cv_mul(in0.cv[1], in1.cv[0], cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs),
        cryptoContext.moduliQ,
        in0.state.cur_limbs,
    )
    axax = F.cv_mul(in0.cv[1], in1.cv[1], cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs)
    sc_factor = cryptoContext.scale_at(in0.state.cur_limbs)
    return in0.cipher_like(
        [bx, ax, axax],
        state=CipherState(
            in0.state.cur_limbs,
            in0.state.noise_deg + in1.state.noise_deg,
            in0.state.scaling_factor * sc_factor,
        ),
    )


def _cipher_add_scalar(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    return in0.cipher_like([
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.state.cur_limbs),
        in0.cv[1],
    ])


def _cipher_add_scalar_inplace(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.state.cur_limbs, inplace=True)
    return in0.replace_with(in0.cipher_like(in0.cv))


def _cipher_sub_scalar(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    return in0.cipher_like([
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.state.cur_limbs),
        in0.cv[1],
    ])


def _cipher_sub_scalar_inplace(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.state.cur_limbs, inplace=True)
    return in0.replace_with(in0.cipher_like(in0.cv))


def _cipher_mul_scalar_double(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    sc_factor = cryptoContext.scale_at(in0.state.cur_limbs)
    if _can_use_pair_kernel(in0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu):
        cv = F.cv_mul_scalar_pair(
            in0.cv[0],
            in0.cv[1],
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.state.cur_limbs,
        )
    else:
        cv = [
            F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs)
            for cv0 in in0.cv
        ]
    return in0.cipher_like(
        cv,
        state=CipherState(
            in0.state.cur_limbs,
            in0.state.noise_deg + 1,
            in0.state.scaling_factor * sc_factor,
        ),
    )


def _cipher_mul_scalar_double_inplace(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    sc_factor = cryptoContext.scale_at(in0.state.cur_limbs)
    if _can_use_pair_kernel(in0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu):
        F.cv_mul_scalar_pair_(
            in0.cv[0],
            in0.cv[1],
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.state.cur_limbs,
        )
    else:
        for cv0 in in0.cv:
            F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs, inplace=True)
    return in0.replace_with(
        in0.cipher_like(
            in0.cv,
            state=CipherState(
                in0.state.cur_limbs,
                in0.state.noise_deg + 1,
                in0.state.scaling_factor * sc_factor,
            ),
        )
    )


def _cipher_mul_scalar_int(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    if _can_use_pair_kernel(in0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu):
        cv = F.cv_mul_scalar_pair(
            in0.cv[0],
            in0.cv[1],
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.state.cur_limbs,
        )
    else:
        cv = [
            F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs)
            for cv0 in in0.cv
        ]
    return in0.cipher_like(cv)


def _cipher_mul_scalar_int_inplace(in0, scalar, cryptoContext):
    scalar_mod = _scalar_tensor(scalar, cryptoContext, in0.state.cur_limbs, in0.cv[0].device)
    if _can_use_pair_kernel(in0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu):
        F.cv_mul_scalar_pair_(
            in0.cv[0],
            in0.cv[1],
            scalar_mod,
            cryptoContext.moduliQ,
            cryptoContext.q_mu,
            in0.state.cur_limbs,
        )
    else:
        for cv0 in in0.cv:
            F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.state.cur_limbs, inplace=True)
    return in0.replace_with(in0.cipher_like(in0.cv))


def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.state.cur_limbs) for cv0 in in0.cv]
    return in0.cipher_like(cv)
