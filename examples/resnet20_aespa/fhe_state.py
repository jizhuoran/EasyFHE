def rescale_one_level(cipher, cryptoContext):
    import easyfhe.fhe as fhe

    return fhe.align_to(
        cipher,
        fhe.CipherState(cipher.state.cur_limbs - 1, cipher.state.noise_deg - 1),
        cryptoContext,
    )


def reduce_noise_to_one(cipher, cryptoContext):
    if cipher.state.noise_deg == 1:
        return cipher
    if cipher.state.noise_deg != 2:
        raise ValueError(f"Expected noise_deg 1 or 2, got {cipher.state.noise_deg}")
    return rescale_one_level(cipher, cryptoContext)


def parse_rotation_key_limb_limits(values):
    limits = {}
    for value in values or ():
        try:
            rotation, limbs = str(value).split(":", 1)
            limits[int(rotation)] = int(limbs)
        except ValueError as exc:
            raise ValueError(
                f"invalid --rot-key-limb-limit {value!r}; expected ROT:LIMBS"
            ) from exc
    return limits
