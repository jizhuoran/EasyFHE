import argparse
import os
import time

import numpy as np

import easyfhe as torch
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe

from examples.resnet20_aespa.cli import parse_rotation_key_limb_limits
from examples.resnet20_aespa.formatting import format_bytes, format_seconds
from examples.resnet20_aespa.main import build_config


def _parse_pair_or_scalar(value):
    if value is None:
        return None
    parts = str(value).replace(",", ":").split(":")
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise ValueError(f"expected INT or C2S:S2C, got {value!r}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ResNet20 AESPA bootstrapping in isolation.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("EASYFHE_DEVICE", "cuda"))
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--auto-load-keys", dest="auto_load_keys", action="store_true", default=None)
    parser.add_argument("--no-auto-load-keys", dest="auto_load_keys", action="store_false")
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
    parser.add_argument(
        "--rotation-random-mode",
        choices=("fresh", "reuse_by_shape"),
        default="fresh",
    )
    parser.add_argument(
        "--rot-key-limb-limit",
        action="append",
        default=[],
        metavar="ROT:LIMBS",
    )
    parser.add_argument("--baby-step", default=None, help="BSGS baby-step override as INT or C2S:S2C")
    parser.add_argument(
        "--constant-cache-mode",
        choices=("none", "plain", "middle", "both"),
        default="both",
    )
    parser.add_argument("--total", type=int, default=1)
    return parser.parse_args()


def _sync(ctx):
    if ctx.device == "cuda":
        torch.cuda.synchronize()


def _format_cache(constants):
    if not hasattr(constants, "cache_info"):
        return "unavailable"
    info = constants.cache_info()
    return (
        f"mode={info['mode']} "
        f"middle={info['middle_entries']}({format_bytes(info['middle_bytes'])}) "
        f"plain={info['plain_entries']}({format_bytes(info['plain_bytes'])}) "
        f"scalar={info['scalar_entries']}({format_bytes(info['scalar_bytes'])}) "
        f"plain_hits={info['plain_hits']} "
        f"plain_misses={info['plain_misses']} "
        f"scalar_hits={info['scalar_hits']} "
        f"scalar_misses={info['scalar_misses']} "
        f"middle_hits={info['middle_hits']} "
        f"middle_misses={info['middle_misses']}"
    )


def _build_bootstrap_runtime(args):
    config = build_config(args)
    baby_step = _parse_pair_or_scalar(args.baby_step)
    bootstrap_extra_depth = bs.depth(
        log_bs_slots=config.log_bs_slots,
        level_budget=config.level_budgets,
        secret_key_dist=config.secret_key_dist,
    )
    bootstrap_rotations = bs.plan_rot_keys(
        log_n=config.log_n,
        log_bs_slots=config.log_bs_slots,
        level_budget=config.level_budgets,
        strategy=config.bootstrap_strategy,
        baby_step=baby_step,
    )
    rotations = tuple(dict.fromkeys([*config.rotate_indices, *bootstrap_rotations]))

    setup_start = time.perf_counter()
    client, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=config.post_bootstrap_levels + bootstrap_extra_depth,
            log_n=config.log_n,
            dnum=config.dnum,
            dcrt_bits=config.dcrt_bits,
            first_mod=config.first_mod,
            secret_key_dist=config.secret_key_dist,
            scale_mode=config.scale_mode,
            rescale_policy=config.rescale_policy,
            rotations=rotations,
            auto_load_keys=args.auto_load_keys,
            rotation_random_mode=str(args.rotation_random_mode),
            rotation_key_limb_limits=parse_rotation_key_limb_limits(args.rot_key_limb_limit),
        ),
        device=config.device,
    )
    setup_seconds = time.perf_counter() - setup_start

    bootstrap_material = {}
    constant_seconds = []
    for log_bs_slots, level_budget in zip(config.log_bs_slots, config.level_budgets):
        constant_start = time.perf_counter()
        constants, plan = bs.generate(
            ctx,
            log_bs_slots=log_bs_slots,
            level_budget=level_budget,
            post_bootstrap_levels=config.post_bootstrap_levels,
            strategy=config.bootstrap_strategy,
            baby_step=baby_step,
        )
        constants.set_cache_mode(args.constant_cache_mode)
        constant_seconds.append(time.perf_counter() - constant_start)
        bootstrap_material[int(log_bs_slots)] = (constants, plan)

    return config, client, ctx, bootstrap_material, setup_seconds, constant_seconds


def _make_cipher(client, ctx, log_bs_slots, seed):
    slots = 1 << int(log_bs_slots)
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.5, 0.5, size=slots).astype(np.double)
    return client.encrypt(values, device=ctx.device, scale_deg=1, level=0, slots=slots)


def _run_timed_bootstrap(ctx, cipher, constants, plan, iters, bootstrap_mode):
    times = []
    out = None
    for _ in range(iters):
        _sync(ctx)
        start = time.perf_counter()
        out = bs.bootstrap(cipher, ctx, constants, plan, L0=cipher.state.cur_limbs, bootstrap_mode=bootstrap_mode)
        _sync(ctx)
        times.append(time.perf_counter() - start)
    return out, times


def _print_timing(times):
    if not times:
        print("timed bootstrap: no iterations")
        return
    avg = sum(times) / len(times)
    print(
        "timed bootstrap:",
        f"iters={len(times)}",
        f"avg={format_seconds(avg)}",
        f"min={format_seconds(min(times))}",
        f"max={format_seconds(max(times))}",
    )
    print("per-iter:", ", ".join(format_seconds(value) for value in times))


def main():
    args = _parse_args()
    if args.iters < 0 or args.warmup < 0:
        raise ValueError("--iters and --warmup must be non-negative")

    config, client, ctx, bootstrap_material, setup_seconds, constant_seconds = _build_bootstrap_runtime(args)
    log_bs_slots = int(config.log_bs_slots[0])
    constants, plan = bootstrap_material[log_bs_slots]
    cipher = _make_cipher(client, ctx, log_bs_slots, args.seed)

    print("================ ResNet20 AESPA bootstrap benchmark ================")
    print(f"device: {ctx.device}")
    print(f"bootstrap_strategy: {plan.strategy}")
    print(f"bootstrap_mode: {config.bootstrap_mode}")
    print(f"baby_step: {plan.baby_step}")
    print(f"log_bs_slots: {log_bs_slots}")
    print(f"cipher: cur_limbs={cipher.state.cur_limbs} noise_deg={cipher.state.noise_deg} slots={cipher.slots}")
    print(f"context setup: {format_seconds(setup_seconds)}")
    print("constant/key setup:", ", ".join(format_seconds(value) for value in constant_seconds))
    print("constant cache before:", _format_cache(constants))

    if args.warmup:
        print(f"warmup: {args.warmup}")
        _run_timed_bootstrap(ctx, cipher, constants, plan, args.warmup, config.bootstrap_mode)
        print("constant cache after warmup:", _format_cache(constants))

    _, times = _run_timed_bootstrap(ctx, cipher, constants, plan, args.iters, config.bootstrap_mode)

    _print_timing(times)
    print("constant cache after:", _format_cache(constants))


if __name__ == "__main__":
    main()
