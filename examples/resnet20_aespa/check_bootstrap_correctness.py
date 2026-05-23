import argparse
import math
import os

import numpy as np

import easyfhe as torch
import easyfhe.bs.openfhe as bs
from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import arithmetic
from easyfhe.fhe.ops import rotation
from easyfhe.bs.openfhe.runtime import approx as bootstrap_approx
from easyfhe.bs.openfhe.runtime.bootstrap import (
    _raise_ciphertext,
    _scale_after_approx,
    _scale_to_original_message,
    eval_coeffs_to_slots,
    eval_slots_to_coeffs,
)

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
    parser.add_argument("--input-level", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--trace-stages", action="store_true")
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


def _decrypt(client, cipher, ctx):
    _sync(ctx)
    return client.decrypt(cipher).cpu().numpy().reshape(-1)


def _print_stage(name, cipher, client, ctx, expected=None, slots=None):
    values = _decrypt(client, cipher, ctx)
    if slots is not None:
        values = values[:slots]
    finite = bool(np.isfinite(values).all())
    print(
        f"[stage] {name}:",
        f"state={cipher.state}",
        f"slots={cipher.slots}",
        f"finite={finite}",
        f"min={float(np.min(values)):.6e}",
        f"max={float(np.max(values)):.6e}",
        f"mean_abs={float(np.mean(np.abs(values))):.6e}",
    )
    if expected is not None:
        stats = _error_stats(values[: len(expected)], expected)
        print(
            f"        vs expected:",
            f"max_abs={stats['max_abs']:.6e}",
            f"mean_abs={stats['mean_abs']:.6e}",
            f"rmse={stats['rmse']:.6e}",
        )
    print("        sample:", np.array2string(values[:8], precision=6, separator=", "))
    return values


def _replicate_sparse_slots(cipher, slots, ctx):
    for step in range(int(math.log2(ctx.N // (2 * slots)))):
        cipher = rotation.homo_rotate_add(
            cipher,
            (1 << step) * slots,
            ctx,
            addend=cipher,
        )
    return cipher.cipher_like(cipher.cv, slots=slots)


def _trace_modraise_first_sparse(cipher, values, client, ctx, constants, plan):
    if getattr(plan, "bootstrap_mode", "modraise_first") != "modraise_first":
        print("[trace] stage trace currently covers modraise_first only")
        return
    if int(plan.slots) == int(ctx.M // 4):
        print("[trace] stage trace currently focuses on sparse bootstrap")
        return

    slots = int(plan.slots)
    print("================ OpenFHE bootstrap stage trace ================")
    _print_stage("input", cipher, client, ctx, expected=values, slots=slots)

    raised = _raise_ciphertext(cipher, ctx, constants, cipher.state.cur_limbs)
    _print_stage("raised", raised, client, ctx, slots=slots)

    replicated = _replicate_sparse_slots(raised, slots, ctx)
    replicated = alignment.reduce_noise_to_one(replicated, ctx)
    _print_stage("sparse_replicated", replicated, client, ctx, slots=slots)

    c2s = eval_coeffs_to_slots(replicated, ctx, constants, plan)
    _print_stage("c2s", c2s, client, ctx, slots=slots)

    c2s_reduced = alignment.reduce_noise_to_one(c2s, ctx)
    _print_stage("c2s_reduced", c2s_reduced, client, ctx, slots=slots)

    s2c = eval_slots_to_coeffs(c2s_reduced, ctx, constants, plan)
    s2c = rotation.homo_rotate_add(s2c, slots, ctx, addend=s2c)
    s2c = s2c.cipher_like(s2c.cv, slots=slots)
    replicated_values = _decrypt(client, replicated, ctx)[:slots]
    _print_stage("c2s_s2c_roundtrip", s2c, client, ctx, expected=replicated_values, slots=slots)

    eval_mod_input = rotation.homo_rotate_add(c2s, 2 * ctx.N - 1, ctx, addend=c2s)
    eval_mod_input = alignment.reduce_noise_to_one(eval_mod_input, ctx)
    _print_stage("eval_mod_input", eval_mod_input, client, ctx, slots=slots)

    approx = bootstrap_approx.eval_bootstrap_approx_mod(eval_mod_input, ctx, constants, plan)
    _print_stage("approx", approx, client, ctx, slots=slots)

    scaled = _scale_after_approx(approx, ctx, constants)
    scaled = alignment.reduce_noise_to_one(scaled, ctx)
    _print_stage("post_approx_scale", scaled, client, ctx, slots=slots)

    decoded = eval_slots_to_coeffs(scaled, ctx, constants, plan)
    decoded = rotation.homo_rotate_add(decoded, slots, ctx, addend=decoded)
    decoded = decoded.cipher_like(decoded.cv, slots=slots)
    decoded = _scale_to_original_message(decoded, ctx, constants)
    decoded = alignment.reduce_noise_to_one(decoded, ctx)
    _print_stage("decoded_scaled", decoded, client, ctx, expected=values, slots=slots)


def main():
    args = _parse_args()
    config, client, ctx, bootstrap_material, _, _ = _build_bootstrap_runtime(args)
    log_bs_slots = int(config.log_bs_slots[0])
    constants, plan = bootstrap_material[log_bs_slots]

    slots = 1 << log_bs_slots
    rng = np.random.default_rng(int(args.seed))
    values = rng.uniform(-float(args.amplitude), float(args.amplitude), size=slots).astype(np.float64)

    cipher = client.encrypt(values, device=ctx.device, scale_deg=1, level=int(args.input_level), slots=slots)
    if args.trace_stages:
        _trace_modraise_first_sparse(cipher, values, client, ctx, constants, plan)

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
