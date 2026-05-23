import argparse
import math
import os

import numpy as np

import easyfhe as torch
import easyfhe.bs.openfhe as bs

try:
    from .benchmark_bootstrap import _build_bootstrap_runtime, _sync
except ImportError:
    from benchmark_bootstrap import _build_bootstrap_runtime, _sync


def _parse_args():
    parser = argparse.ArgumentParser(description="Check OpenFHE bootstrap decrypt correctness.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("EASYFHE_DEVICE", "cuda"))
    parser.add_argument(
        "--bootstrap-strategy",
        choices=("double_hoist", "normal_giant", "normal_bsgs"),
        default=os.environ.get("EASYFHE_BOOTSTRAP_STRATEGY", "double_hoist"),
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=("classic", "modraise_first", "slots_first", "stc_first"),
        default=os.environ.get("EASYFHE_BOOTSTRAP_MODE", "modraise_first"),
    )
    parser.add_argument(
        "--secret-key-dist",
        choices=("SPARSE_TERNARY", "UNIFORM_TERNARY"),
        default=os.environ.get("EASYFHE_SECRET_KEY_DIST", "SPARSE_TERNARY"),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amplitude", type=float, default=0.45)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--auto-load-keys", dest="auto_load_keys", action="store_true", default=None)
    parser.add_argument("--no-auto-load-keys", dest="auto_load_keys", action="store_false")
    parser.add_argument("--rotation-random-mode", choices=("fresh", "reuse_by_shape"), default="fresh")
    parser.add_argument("--rot-key-limb-limit", action="append", default=[], metavar="ROT:LIMBS")
    parser.add_argument("--total", type=int, default=1)
    return parser.parse_args()


def _error_stats(actual, expected):
    error = np.asarray(actual) - np.asarray(expected)
    abs_error = np.abs(error)
    return {
        "max_abs": float(np.max(abs_error)),
        "mean_abs": float(np.mean(abs_error)),
        "rmse": float(math.sqrt(float(np.mean(error * error)))),
    }


def main():
    args = _parse_args()
    config, client, ctx, bootstrap_material, _, _ = _build_bootstrap_runtime(args)
    log_bs_slots = int(config.log_bs_slots[0])
    constants, plan = bootstrap_material[log_bs_slots]

    slots = 1 << log_bs_slots
    rng = np.random.default_rng(int(args.seed))
    values = rng.uniform(-float(args.amplitude), float(args.amplitude), size=slots).astype(np.float64)

    cipher = client.encrypt(values, device=ctx.device, scale_deg=1, level=0, slots=slots)
    _sync(ctx)
    output = bs.bootstrap(cipher, ctx, constants, plan, L0=cipher.state.cur_limbs)
    _sync(ctx)

    decrypted = client.decrypt(output).cpu().numpy().reshape(-1)[:slots]
    stats = _error_stats(decrypted, values)
    ok = np.allclose(decrypted, values, rtol=float(args.rtol), atol=float(args.atol))

    print("================ OpenFHE bootstrap correctness ================")
    print(f"device: {ctx.device}")
    print(f"bootstrap_strategy: {plan.strategy}")
    print(f"bootstrap_mode: {getattr(plan, 'bootstrap_mode', 'modraise_first')}")
    print(f"secret_key_dist: {config.secret_key_dist}")
    print(f"log_bs_slots: {log_bs_slots}")
    print(f"input_state: {cipher.state}")
    print(f"output_state: {output.state}")
    print(f"max_abs={stats['max_abs']:.6e} mean_abs={stats['mean_abs']:.6e} rmse={stats['rmse']:.6e}")
    print("sample expected:", values[:8])
    print("sample actual:  ", decrypted[:8])
    print("result:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
