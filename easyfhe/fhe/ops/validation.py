import math

from ..ciphertext import Cipher


def validate_cipher_op(
    op_name,
    cipher,
    *,
    require_ext=None,
    require_scale_degree=None,
    require_components=None,
):
    if require_ext is not None and cipher.is_ext != require_ext:
        raise ValueError(f"{op_name}: expected is_ext={require_ext}, got {cipher.is_ext}")
    if require_scale_degree is not None and cipher.state.scale_degree != require_scale_degree:
        raise ValueError(f"{op_name}: cipher scale_degree must be {require_scale_degree}, got {cipher.state.scale_degree}")
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
    require_scale_degree=None,
    require_same_metadata=(),
):
    if cipher.is_ext != plaintext.is_ext:
        raise ValueError(f"{op_name}: is_ext mismatch: {cipher.is_ext} != {plaintext.is_ext}")
    if require_ext is not None and cipher.is_ext != require_ext:
        raise ValueError(f"{op_name}: expected is_ext={require_ext}, got {cipher.is_ext}")
    if require_scale_degree is not None:
        if cipher.state.scale_degree != require_scale_degree:
            raise ValueError(f"{op_name}: cipher scale_degree must be {require_scale_degree}, got {cipher.state.scale_degree}")
        if plaintext.state.scale_degree != require_scale_degree:
            raise ValueError(f"{op_name}: plaintext scale_degree must be {require_scale_degree}, got {plaintext.state.scale_degree}")
    validate_matching_metadata(op_name, cipher, plaintext, require_same_metadata)


def validate_component_count(op_name, value, *, expected):
    if len(value.cv) != expected:
        raise ValueError(f"{op_name}: expected {expected} components, got {len(value.cv)}")


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


def validate_slot_count(op_name, slots, context):
    slots = validate_positive_int(op_name, "slots", slots)
    if slots & (slots - 1):
        raise ValueError(f"{op_name}: slots must be a power of two, got {slots}")
    max_slots = int(context.N) // 2
    if slots > max_slots:
        raise ValueError(f"{op_name}: slots [{slots}] exceeds max slots [{max_slots}]")
    return slots


def validate_matching_metadata(op_name, left, right, fields):
    for field in fields:
        left_value = _metadata_value(left, field)
        right_value = _metadata_value(right, field)
        if not _metadata_matches(field, left_value, right_value):
            raise ValueError(_metadata_mismatch_message(op_name, field, left_value, right_value))


def _metadata_value(value, field):
    if field in ("cur_limbs", "scale_degree", "scaling_factor"):
        return getattr(value.state, field)
    return getattr(value, field)


def _metadata_matches(field, left_value, right_value):
    if field != "scaling_factor":
        return left_value == right_value
    if left_value is None or right_value is None:
        return left_value is right_value
    return math.isclose(float(left_value), float(right_value), rel_tol=1e-12, abs_tol=0.0)


def _metadata_mismatch_message(op_name, field, left_value, right_value):
    if field != "scaling_factor" or left_value is None or right_value is None:
        return f"{op_name}: {field} mismatch: {left_value} != {right_value}"

    left = float(left_value)
    right = float(right_value)
    rel = abs(left - right) / max(abs(left), abs(right))
    return (
        f"{op_name}: {field} mismatch: {left_value} != {right_value} "
        f"(log2 {math.log2(left):.12f} != {math.log2(right):.12f}, rel_delta={rel:.3e})"
    )
