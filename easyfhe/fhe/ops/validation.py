from ..ciphertext import Cipher


def validate_cipher_op(
    op_name,
    cipher,
    *,
    require_ext=None,
    require_noise_deg=None,
    require_components=None,
):
    if require_ext is not None and cipher.is_ext != require_ext:
        raise ValueError(f"{op_name}: expected is_ext={require_ext}, got {cipher.is_ext}")
    if require_noise_deg is not None and cipher.state.noise_deg != require_noise_deg:
        raise ValueError(f"{op_name}: cipher noise_deg must be {require_noise_deg}, got {cipher.state.noise_deg}")
    if require_components is not None:
        validate_component_count(op_name, cipher, expected=require_components)


def validate_binary_cipher_op(
    op_name,
    left,
    right,
    *,
    require_ext=None,
    require_components=None,
    require_same_metadata=(),
):
    if left.is_ext != right.is_ext:
        raise ValueError(f"{op_name}: is_ext mismatch: {left.is_ext} != {right.is_ext}")
    if require_ext is not None and left.is_ext != require_ext:
        raise ValueError(f"{op_name}: expected is_ext={require_ext}, got {left.is_ext}")
    if require_components is not None:
        validate_component_count(op_name, left, expected=require_components)
        validate_component_count(op_name, right, expected=require_components)
    validate_matching_metadata(op_name, left, right, require_same_metadata)


def validate_cipher_plain_op(
    op_name,
    cipher,
    plaintext,
    *,
    require_ext=None,
    require_noise_deg=None,
    require_same_metadata=(),
):
    if cipher.is_ext != plaintext.is_ext:
        raise ValueError(f"{op_name}: is_ext mismatch: {cipher.is_ext} != {plaintext.is_ext}")
    if require_ext is not None and cipher.is_ext != require_ext:
        raise ValueError(f"{op_name}: expected is_ext={require_ext}, got {cipher.is_ext}")
    if require_noise_deg is not None:
        if cipher.state.noise_deg != require_noise_deg:
            raise ValueError(f"{op_name}: cipher noise_deg must be {require_noise_deg}, got {cipher.state.noise_deg}")
        if plaintext.state.noise_deg != require_noise_deg:
            raise ValueError(f"{op_name}: plaintext noise_deg must be {require_noise_deg}, got {plaintext.state.noise_deg}")
    validate_matching_metadata(op_name, cipher, plaintext, require_same_metadata)


def validate_component_count(op_name, value, *, expected):
    if len(value.cv) != expected:
        raise ValueError(f"{op_name}: expected {expected} components, got {len(value.cv)}")


def require_encoded_scalar(value, op_name):
    if hasattr(value, "to") and hasattr(value, "dim"):
        return value
    if isinstance(value, (list, tuple)):
        return value
    raise TypeError(
        f"{op_name}: expected an encoded scalar tensor or per-limb scalar list; "
        "encode raw constants with ConstantBundle.encoded_scalars or arithmetic._encode_*_for_scalar_op"
    )


def require_batched_cipher(op_name, value, arg_name):
    if not isinstance(value, Cipher):
        raise TypeError(f"{arg_name}: expected a batched Cipher, got {type(value)}")
    if value.batch_size <= 0:
        raise ValueError(f"{op_name}: expected at least one cipher")


def validate_positive_int(op_name, name, value):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{op_name}: {name} must be positive, got {value}")
    return value


def validate_slot_count(op_name, slots, cryptoContext):
    slots = validate_positive_int(op_name, "slots", slots)
    if slots & (slots - 1):
        raise ValueError(f"{op_name}: slots must be a power of two, got {slots}")
    max_slots = int(cryptoContext.N) // 2
    if slots > max_slots:
        raise ValueError(f"{op_name}: slots [{slots}] exceeds max slots [{max_slots}]")
    return slots


def validate_matching_metadata(op_name, left, right, fields):
    for field in fields:
        left_value = _metadata_value(left, field)
        right_value = _metadata_value(right, field)
        if left_value != right_value:
            raise ValueError(f"{op_name}: {field} mismatch: {left_value} != {right_value}")


def _metadata_value(value, field):
    if field in ("cur_limbs", "noise_deg", "scaling_factor"):
        return getattr(value.state, field)
    return getattr(value, field)
