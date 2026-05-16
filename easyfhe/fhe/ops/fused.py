from ..runtime.instrumentation import run_instrumented_op
from .arithmetic import homo_add
from .plaintext import homo_mul_pt


def fused_broadcast_mac(cipher, plaintexts, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "fused_broadcast_mac",
        _fused_broadcast_mac,
        cipher,
        plaintexts,
        cryptoContext,
    )


def _fused_broadcast_mac(cipher, plaintexts, cryptoContext):
    plaintexts = tuple(plaintexts)
    if not plaintexts:
        raise ValueError("fused_broadcast_mac: expected at least one plaintext")

    total = homo_mul_pt(cipher, plaintexts[0], cryptoContext)
    for plaintext in plaintexts[1:]:
        total = homo_add(total, homo_mul_pt(cipher, plaintext, cryptoContext), cryptoContext)
    return total


def fused_pairwise_mac(ciphers, plaintexts, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "fused_pairwise_mac",
        _fused_pairwise_mac,
        ciphers,
        plaintexts,
        cryptoContext,
    )


def _fused_pairwise_mac(ciphers, plaintexts, cryptoContext):
    ciphers = tuple(ciphers)
    plaintexts = tuple(plaintexts)
    if not ciphers:
        raise ValueError("fused_pairwise_mac: expected at least one cipher/plaintext pair")
    if len(ciphers) != len(plaintexts):
        raise ValueError(
            "fused_pairwise_mac: cipher and plaintext lengths must match, "
            f"got {len(ciphers)} and {len(plaintexts)}"
        )

    total = homo_mul_pt(ciphers[0], plaintexts[0], cryptoContext)
    for cipher, plaintext in zip(ciphers[1:], plaintexts[1:]):
        total = homo_add(total, homo_mul_pt(cipher, plaintext, cryptoContext), cryptoContext)
    return total
