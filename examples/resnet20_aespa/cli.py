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
