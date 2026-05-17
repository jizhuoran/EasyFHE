import argparse
import os
import time
from types import SimpleNamespace

import numpy as np

import easyfhe as torch
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe
from easyfhe.fhe.runtime.instrumentation import profile

try:
    from .fhe_state import runtime_options_from_args
    from .main import build_config, _format_seconds, _format_bytes
except ImportError:
    from fhe_state import runtime_options_from_args
    from main import build_config, _format_seconds, _format_bytes


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ResNet20 AESPA bootstrapping in isolation.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("EASYFHE_DEVICE", "cuda"))
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", "--time-ops", dest="time_ops", action="store_true")
    parser.add_argument(
        "--profile-detail",
        choices=("phase", "c2s-fastrot", "s2c", "s2c-fastrot", "fastrot", "fastrot-inner", "all"),
        default="phase",
    )
    parser.add_argument("--profile-limit", type=int, default=32)
    parser.add_argument("--auto-sync", action="store_true")
    parser.add_argument("--count-ops", action="store_true")
    parser.add_argument("--auto-load-keys", dest="auto_load_keys", action="store_true", default=None)
    parser.add_argument("--no-auto-load-keys", dest="auto_load_keys", action="store_false")
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
        f"middle={info['middle_entries']}({_format_bytes(info['middle_bytes'])}) "
        f"plain={info['plain_entries']}({_format_bytes(info['plain_bytes'])}) "
        f"plain_batch={info['plain_batch_entries']}({_format_bytes(info['plain_batch_bytes'])}) "
        f"plain_hits={info['plain_hits']} "
        f"plain_misses={info['plain_misses']} "
        f"plain_batch_hits={info['plain_batch_hits']} "
        f"plain_batch_misses={info['plain_batch_misses']} "
        f"middle_hits={info['middle_hits']} "
        f"middle_misses={info['middle_misses']}"
    )


def _build_bootstrap_runtime(args):
    config = build_config(args)
    # Keep setup quiet; the benchmark installs a scoped profiler around only
    # the measured bootstrap loop.
    options_args = SimpleNamespace(**vars(args))
    options_args.count_ops = False
    options_args.time_ops = False
    options_args.auto_sync = False
    options = runtime_options_from_args(options_args)
    bootstrap_extra_depth = bs.depth(
        log_bs_slots=config.log_bs_slots,
        level_budget=config.level_budgets,
        secret_key_dist=config.secret_key_dist,
    )

    setup_start = time.perf_counter()
    ctx = fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=config.max_levels_remaining + bootstrap_extra_depth,
            log_n=config.log_n,
            dnum=config.dnum,
            dcrt_bits=config.dcrt_bits,
            first_mod=config.first_mod,
            secret_key_dist=config.secret_key_dist,
            rescale_tech=config.rescale_tech,
            rotations=config.rotate_indices,
        ),
        device=config.device,
        options=options,
    )
    setup_seconds = time.perf_counter() - setup_start

    constants_by_slots = {}
    constant_seconds = []
    for log_bs_slots, level_budget in zip(config.log_bs_slots, config.level_budgets):
        constant_start = time.perf_counter()
        bs_keys, constants = bs.generate(
            ctx,
            log_bs_slots=log_bs_slots,
            level_budget=level_budget,
            max_levels_remaining=config.max_levels_remaining,
        )
        ctx.addkeys(bs_keys)
        constant_seconds.append(time.perf_counter() - constant_start)
        constants_by_slots[int(log_bs_slots)] = constants

    return config, ctx, constants_by_slots, setup_seconds, constant_seconds


def _make_cipher(ctx, log_bs_slots, seed):
    slots = 1 << int(log_bs_slots)
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.5, 0.5, size=slots).astype(np.double)
    return ctx.encrypt(values, ctx.device, 1, 0, slots)


def _run_timed_bootstrap(ctx, cipher, constants, iters):
    times = []
    out = None
    for _ in range(iters):
        _sync(ctx)
        start = time.perf_counter()
        out = bs.bootstrap(cipher, ctx, constants, L0=cipher.cur_limbs)
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
        f"avg={_format_seconds(avg)}",
        f"min={_format_seconds(min(times))}",
        f"max={_format_seconds(max(times))}",
    )
    print("per-iter:", ", ".join(_format_seconds(value) for value in times))


def _print_bootstrap_phase_summary(profiler):
    records = profiler.records
    total = records.get("homo_bootstrap")
    if total is None or total.total_time <= 0:
        return

    phases = [
        ("c2s", records.get("bs_c2s")),
        ("eval", records.get("bs_eval_mod")),
        ("s2c", records.get("bs_s2c")),
    ]
    phase_total = sum(record.total_time for _, record in phases if record is not None)
    other = max(0.0, total.total_time - phase_total)

    print("\nBootstrap Phase Breakdown:")
    print(f"{'phase':16s} {'count':>8s} {'total(s)':>12s} {'share':>8s} {'avg(ms)':>12s}")
    for name, record in phases:
        if record is None:
            count = 0
            elapsed = 0.0
            avg = 0.0
        else:
            count = record.count
            elapsed = record.total_time
            avg = record.avg_time
        share = 100.0 * elapsed / total.total_time
        print(f"{name:16s} {count:8d} {elapsed:12.6f} {share:7.2f}% {avg * 1000:12.3f}")
    share = 100.0 * other / total.total_time
    print(f"{'other':16s} {'-':>8s} {other:12.6f} {share:7.2f}% {'-':>12s}")
    print(f"{'total':16s} {total.count:8d} {total.total_time:12.6f} {100.0:7.2f}% {total.avg_time * 1000:12.3f}")


def main():
    args = _parse_args()
    if args.iters < 0 or args.warmup < 0:
        raise ValueError("--iters and --warmup must be non-negative")

    config, ctx, constants_by_slots, setup_seconds, constant_seconds = _build_bootstrap_runtime(args)
    log_bs_slots = int(config.log_bs_slots[0])
    constants = constants_by_slots[log_bs_slots]
    cipher = _make_cipher(ctx, log_bs_slots, args.seed)

    print("================ ResNet20 AESPA bootstrap benchmark ================")
    print(f"device: {ctx.device}")
    print(f"log_bs_slots: {log_bs_slots}")
    print(f"cipher: cur_limbs={cipher.cur_limbs} noise_deg={cipher.noise_deg} slots={cipher.slots}")
    print(f"context setup: {_format_seconds(setup_seconds)}")
    print("constant/key setup:", ", ".join(_format_seconds(value) for value in constant_seconds))
    print("constant cache before:", _format_cache(constants))

    if args.warmup:
        print(f"warmup: {args.warmup}")
        _run_timed_bootstrap(ctx, cipher, constants, args.warmup)
        print("constant cache after warmup:", _format_cache(constants))

    if args.time_ops:
        include = None
        if args.profile_detail == "phase":
            include = {"homo_bootstrap", "bs_c2s", "bs_eval_mod", "bs_s2c"}
        elif args.profile_detail == "s2c":
            include = {
                "homo_bootstrap",
                "bs_s2c",
                "bs_s2c_fast_rotate_ext_batch",
                "bs_s2c_fused_grouped_pairwise_mac",
                "bs_s2c_fused_pairwise_mac",
                "bs_s2c_double_hoist_rotate_sum",
            }
        elif args.profile_detail == "c2s-fastrot":
            include = {
                "homo_bootstrap",
                "bs_c2s",
                "bs_c2s_fast_rotate_ext_batch",
                "bs_c2s_fast_rotate_ext_batch_modup",
                "bs_c2s_fast_rotate_ext_batch_key_products",
                "bs_c2s_fast_rotate_ext_batch_scale_pc",
                "bs_c2s_fast_rotate_ext_batch_precompute_maps",
                "bs_c2s_fast_rotate_ext_batch_finalize",
            }
        elif args.profile_detail == "s2c-fastrot":
            include = {
                "homo_bootstrap",
                "bs_s2c",
                "bs_s2c_fast_rotate_ext_batch",
                "bs_s2c_fast_rotate_ext_batch_modup",
                "bs_s2c_fast_rotate_ext_batch_key_products",
                "bs_s2c_fast_rotate_ext_batch_scale_pc",
                "bs_s2c_fast_rotate_ext_batch_precompute_maps",
                "bs_s2c_fast_rotate_ext_batch_finalize",
            }
        elif args.profile_detail == "fastrot":
            include = {
                "homo_bootstrap",
                "bs_c2s",
                "bs_s2c",
                "bs_c2s_fast_rotate_ext_batch",
                "bs_s2c_fast_rotate_ext_batch",
                "fast_rotate_ext_batch",
                "bs_c2s_fast_rotate_ext_batch_modup",
                "bs_c2s_fast_rotate_ext_batch_key_products",
                "bs_c2s_fast_rotate_ext_batch_scale_pc",
                "bs_c2s_fast_rotate_ext_batch_precompute_maps",
                "bs_c2s_fast_rotate_ext_batch_finalize",
                "bs_s2c_fast_rotate_ext_batch_modup",
                "bs_s2c_fast_rotate_ext_batch_key_products",
                "bs_s2c_fast_rotate_ext_batch_scale_pc",
                "bs_s2c_fast_rotate_ext_batch_precompute_maps",
                "bs_s2c_fast_rotate_ext_batch_finalize",
            }
        elif args.profile_detail == "fastrot-inner":
            include = {
                "homo_bootstrap",
                "fast_rotate_ext_batch_modup",
                "fast_rotate_ext_batch_key_products",
                "fast_rotate_ext_batch_scale_pc",
                "fast_rotate_ext_batch_precompute_maps",
                "fast_rotate_ext_batch_finalize",
            }
        with profile(ctx, sync=bool(args.auto_sync), include=include) as profiler:
            _, times = _run_timed_bootstrap(ctx, cipher, constants, args.iters)
        _print_bootstrap_phase_summary(profiler)
        profiler.print_summary(limit=args.profile_limit)
    else:
        _, times = _run_timed_bootstrap(ctx, cipher, constants, args.iters)

    _print_timing(times)
    print("constant cache after:", _format_cache(constants))


if __name__ == "__main__":
    main()
