import easyfhe.fhe as fhe


def rescale_one_level(cipher, cryptoContext):
    return fhe.align_to(
        cipher,
        fhe.CipherState(cipher.cur_limbs - 1, cipher.noise_deg - 1),
        cryptoContext,
    )


def reduce_noise_to_one(cipher, cryptoContext):
    if cipher.noise_deg == 1:
        return cipher
    if cipher.noise_deg != 2:
        raise ValueError(f"Expected noise_deg 1 or 2, got {cipher.noise_deg}")
    return rescale_one_level(cipher, cryptoContext)


def runtime_options_from_args(args):
    return fhe.RuntimeOptions(
        auto_load_keys=args.auto_load_keys,
        auto_sync=bool(args.auto_sync),
        time_ops=bool(args.time_ops),
        count_ops=bool(args.count_ops),
        rotation_random_mode=str(args.rotation_random_mode),
        rotation_key_limb_limits=_parse_rotation_key_limb_limits(args.rot_key_limb_limit),
    )


def _parse_rotation_key_limb_limits(values):
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
