import argparse
import time

import numpy as np

import easyfhe.fhe as fhe
from easyfhe.fhe.ops import rotation


def _diagonal_values(matrix, offset):
    rows = np.arange(matrix.shape[0])
    cols = (rows + int(offset)) % matrix.shape[1]
    return matrix[rows, cols]


def _plaintext(values, crypto_context, *, level, slots, is_ext=False):
    return fhe.ConstantBundle(vectors={"pt": values}, cache_mode="none").plaintext(
        "pt",
        level,
        slots,
        crypto_context,
        is_ext=is_ext,
    )


def encrypted_bsgs_matvec(cipher, matrix, baby_step, crypto_context):
    slots = int(matrix.shape[0])
    if matrix.shape != (slots, slots):
        raise ValueError("matrix must be square")
    if slots % baby_step != 0:
        raise ValueError("slots must be divisible by baby_step")

    giant_count = slots // baby_step
    baby_offsets = list(range(baby_step))
    giant_offsets = [giant * baby_step for giant in range(giant_count)]
    baby_exts = fhe.fast_rotate(cipher, baby_offsets, crypto_context, output_ext=True)

    inner_exts = []
    for giant_offset in giant_offsets:
        plaintexts = []
        for baby_offset in baby_offsets:
            diagonal = _diagonal_values(matrix, giant_offset + baby_offset)
            plaintext_values = np.roll(diagonal, giant_offset)
            plaintext = _plaintext(
                plaintext_values,
                crypto_context,
                level=crypto_context.L - baby_exts.cur_limbs,
                slots=slots,
                is_ext=True,
            )
            plaintexts.append(plaintext)
        inner_exts.append(
            fhe.fused_grouped_pairwise_mac(
                baby_exts,
                rotation._pack_ciphers(plaintexts),
                1,
                crypto_context,
            )[0]
        )

    return fhe.giant_rotate_sum(inner_exts, baby_step, crypto_context, strategy="ext_double_hoist")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--baby-step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=3e-3)
    parser.add_argument("--rtol", type=float, default=3e-3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    slots = int(args.slots)
    baby_step = int(args.baby_step)
    giant_offsets = [giant * baby_step for giant in range(slots // baby_step)]
    rotations = tuple(sorted(set(range(1, baby_step)) | {offset for offset in giant_offsets if offset}))

    crypto_context = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=14,
            dnum=2,
            dcrt_bits=40,
            first_mod=45,
            rotations=rotations,
        ),
        device=args.device,
    )

    matrix = rng.uniform(-0.01, 0.01, size=(slots, slots)).astype(np.double)
    vector = rng.uniform(-0.5, 0.5, size=slots).astype(np.double)
    baseline = matrix @ vector

    cipher = crypto_context.encrypt(vector, crypto_context.device, 1, 0, slots)

    start = time.perf_counter()
    result = encrypted_bsgs_matvec(cipher, matrix, baby_step, crypto_context)
    elapsed = time.perf_counter() - start

    decrypted = crypto_context.decrypt(result).cpu().numpy().reshape(-1)[:slots]
    diff = decrypted - baseline
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(baseline), 1e-12))

    print(f"slots={slots} baby_step={baby_step} giant_count={slots // baby_step}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"max_abs={max_abs:.6e}")
    print(f"rmse={rmse:.6e}")
    print(f"rel_l2={rel_l2:.6e}")
    print("sample decrypted=", np.array2string(decrypted[:8], precision=6, separator=", "))
    print("sample baseline= ", np.array2string(baseline[:8], precision=6, separator=", "))

    np.testing.assert_allclose(decrypted, baseline, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    main()
