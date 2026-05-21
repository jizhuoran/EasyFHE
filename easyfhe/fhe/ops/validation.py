def validate_cipher_op(
    op_name,
    cipher,
    *,
    require_ext=None,
    require_noise_deg=None,
):
    if require_ext is not None and cipher.is_ext != require_ext:
        raise ValueError(f"{op_name}: expected is_ext={require_ext}, got {cipher.is_ext}")
    if require_noise_deg is not None and cipher.noise_deg != require_noise_deg:
        raise ValueError(f"{op_name}: cipher noise_deg must be {require_noise_deg}, got {cipher.noise_deg}")


def validate_binary_cipher_op(op_name, left, right, *, require_ext=None, require_same_metadata=()):
    if left.is_ext != right.is_ext:
        raise ValueError(f"{op_name}: is_ext mismatch: {left.is_ext} != {right.is_ext}")
    if require_ext is not None and left.is_ext != require_ext:
        raise ValueError(f"{op_name}: expected is_ext={require_ext}, got {left.is_ext}")
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
        if cipher.noise_deg != require_noise_deg:
            raise ValueError(f"{op_name}: cipher noise_deg must be {require_noise_deg}, got {cipher.noise_deg}")
        if plaintext.noise_deg != require_noise_deg:
            raise ValueError(f"{op_name}: plaintext noise_deg must be {require_noise_deg}, got {plaintext.noise_deg}")
    validate_matching_metadata(op_name, cipher, plaintext, require_same_metadata)


def validate_cipher_scalar_op(
    op_name,
    cipher,
    *,
    require_ext=None,
    require_noise_deg=None,
):
    validate_cipher_op(
        op_name,
        cipher,
        require_ext=require_ext,
        require_noise_deg=require_noise_deg,
    )


def validate_matching_metadata(op_name, left, right, fields):
    for field in fields:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value != right_value:
            raise ValueError(f"{op_name}: {field} mismatch: {left_value} != {right_value}")
