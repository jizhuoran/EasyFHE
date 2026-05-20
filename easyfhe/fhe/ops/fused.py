from ..ciphertext import Cipher
from . import kernels as F


def fused_broadcast_mac(cipher, plaintexts, cryptoContext):
    plaintext_batch = _require_cipher(plaintexts, "plaintexts")
    if _batch_size(plaintext_batch) == 0:
        raise ValueError("fused_broadcast_mac: expected at least one plaintext")
    _validate_cipher_plain_batch("fused_broadcast_mac", cipher, plaintext_batch)
    cv = F.cipher_fused_broadcast_mac(cipher, plaintext_batch, cryptoContext)
    return _mac_result_like(cipher, plaintext_batch, cv, batch_size=1)


def fused_grouped_pairwise_mac(ciphers, plaintexts, groups, cryptoContext):
    # TODO(perf): The grouped CUDA kernel is currently slower than the old
    # per-group pairwise_mac loop in bootstrapping c2s/s2c. Revisit the kernel
    # layout and compare against a fallback loop before relying on it broadly.
    cipher_batch = _require_cipher(ciphers, "ciphers")
    plaintext_batch = _require_cipher(plaintexts, "plaintexts")
    groups = int(groups)
    if groups <= 0:
        raise ValueError(f"fused_grouped_pairwise_mac: groups must be positive, got {groups}")
    if _batch_size(cipher_batch) == 0:
        raise ValueError("fused_grouped_pairwise_mac: expected at least one cipher per group")
    expected_plaintexts = groups * _batch_size(cipher_batch)
    if _batch_size(plaintext_batch) != expected_plaintexts:
        raise ValueError(
            "fused_grouped_pairwise_mac: plaintext batch size must equal groups * cipher batch size, "
            f"got {_batch_size(plaintext_batch)} != {groups} * {_batch_size(cipher_batch)}"
        )
    _validate_cipher_plain_batch("fused_grouped_pairwise_mac", cipher_batch, plaintext_batch)
    cv = F.cipher_fused_grouped_pairwise_mac(cipher_batch, plaintext_batch, groups, cryptoContext)
    return _mac_result_like(cipher_batch, plaintext_batch, cv, batch_size=groups)


def _require_cipher(value, name):
    if not isinstance(value, Cipher):
        raise TypeError(f"{name}: expected a batched Cipher, got {type(value)}")
    return value


def _batch_size(cipher):
    if hasattr(cipher, "batch_size"):
        return int(cipher.batch_size)
    if cipher.cv[0].dim() == 3:
        return int(cipher.cv[0].shape[0])
    return 1


def _validate_cipher_plain_batch(op_name, cipher, plaintext):
    if cipher.is_ext != plaintext.is_ext:
        raise ValueError(f"{op_name}: is_ext mismatch: {cipher.is_ext} != {plaintext.is_ext}")
    for field in ("cur_limbs", "scaling_factor", "slots"):
        if getattr(cipher, field) != getattr(plaintext, field):
            raise ValueError(
                f"{op_name}: {field} mismatch: "
                f"{getattr(cipher, field)} != {getattr(plaintext, field)}"
            )
    if cipher.noise_deg != 1:
        raise ValueError(f"{op_name}: cipher noise_deg must be 1, got {cipher.noise_deg}")
    if plaintext.noise_deg != 1:
        raise ValueError(f"{op_name}: plaintext noise_deg must be 1, got {plaintext.noise_deg}")


def _mac_result_like(cipher, plaintext, cv, batch_size):
    return cipher.cipher_like(
        list(cv),
        scaling_factor=cipher.scaling_factor * plaintext.scaling_factor,
        noise_deg=cipher.noise_deg + plaintext.noise_deg,
        batch_size=batch_size,
        cipher_id="assign",
    )

