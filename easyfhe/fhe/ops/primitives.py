from . import kernels as F


def _cipher_add(in0, in1, cryptoContext):
    cv = [F.cv_add(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
    return in0.cipher_like(cv)


def _cipher_add_ext(in0, in1, cryptoContext):
    cv = [
        F.cv_add(
            cv0,
            cv1,
            cryptoContext.QplusP_map[in0.cur_limbs],
            in0.cur_limbs + cryptoContext.K,
        )
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


def _cipher_sub(in0, in1, cryptoContext):
    cv = [F.cv_sub(cv0, cv1, cryptoContext.moduliQ, in0.cur_limbs) for cv0, cv1 in zip(in0.cv, in1.cv)]
    return in0.cipher_like(cv)


def _cipher_sub_ext(in0, in1, cryptoContext):
    cv = [
        F.cv_sub(
            cv0,
            cv1,
            cryptoContext.QplusP_map[in0.cur_limbs],
            in0.cur_limbs + cryptoContext.K,
        )
        for cv0, cv1 in zip(in0.cv, in1.cv)
    ]
    return in0.cipher_like(cv)


def _cipher_add_plain(cipher, plaintext, cryptoContext):
    moduli = cryptoContext.QplusP_map[cipher.cur_limbs]
    active_limbs = cipher.cur_limbs + (cryptoContext.K if cipher.is_ext else 0)
    cv = [
        F.cv_add(cipher.cv[0], plaintext.cv[0], moduli, active_limbs),
        cipher.cv[1],
    ]
    return cipher.cipher_like(cv)


def _cipher_mul_plain(cipher, plaintext, cryptoContext):
    moduli = cryptoContext.QplusP_map[cipher.cur_limbs]
    mu = cryptoContext.QmuplusPmu_map[cipher.cur_limbs]
    active_limbs = cipher.cur_limbs + (cryptoContext.K if cipher.is_ext else 0)
    plaintext_values = plaintext.cv[0]
    cv0 = F.cv_mul(cipher.cv[0], plaintext_values, moduli, mu, active_limbs)
    cv1 = F.cv_mul(cipher.cv[1], plaintext_values, moduli, mu, active_limbs)
    return cipher.cipher_like(
        [cv0, cv1],
        scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
        noise_deg=cipher.noise_deg + plaintext.noise_deg,
    )


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


def _cipher_add_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    scalar_mod = scalar_mod.to(in0.cv[0].device)
    return in0.cipher_like([
        F.cv_add_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ])


def _cipher_sub_scalar(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    scalar_mod = scalar_mod.to(in0.cv[0].device)
    return in0.cipher_like([
        F.cv_sub_scalar(in0.cv[0], scalar_mod, cryptoContext.moduliQ, in0.cur_limbs),
        in0.cv[1],
    ])


def _cipher_mul_scalar_double(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    scalar_mod = scalar_mod.to(in0.cv[0].device)
    cv = [
        F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
        for cv0 in in0.cv
    ]
    sc_factor = cryptoContext.scale_at(in0.cur_limbs)
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor * sc_factor, noise_deg=in0.noise_deg + 1)


def _cipher_mul_scalar_int(in0, scalar, cryptoContext):
    scalar_mod = F.gen_scalar_tensor(scalar, cryptoContext.moduliQ_scalar, in0.cur_limbs)
    scalar_mod = scalar_mod.to(in0.cv[0].device)
    cv = [
        F.cv_mul_scalar(cv0, scalar_mod, cryptoContext.moduliQ, cryptoContext.q_mu, in0.cur_limbs)
        for cv0 in in0.cv
    ]
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)


def _cipher_neg(in0, cryptoContext):
    cv = [F.cv_neg(cv0, cryptoContext.moduliQ, in0.cur_limbs) for cv0 in in0.cv]
    return in0.cipher_like(cv, scaling_factor=in0.scaling_factor, noise_deg=in0.noise_deg)
