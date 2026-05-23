import argparse
import os
import time
from contextlib import contextmanager

import easyfhe as torch
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe

try:
    from .data import read_image
    from .fhe_state import parse_rotation_key_limb_limits
    from .main import build_config, _format_seconds, _format_bytes
    from .model import AespaRuntime
    from .weight_pack import WeightPack
    from . import model
except ImportError:
    from data import read_image
    from fhe_state import parse_rotation_key_limb_limits
    from main import build_config, _format_seconds, _format_bytes
    from model import AespaRuntime
    from weight_pack import WeightPack
    import model


def _parse_args():
    parser = argparse.ArgumentParser(description="Profile ResNet20 AESPA inference by layer.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("EASYFHE_DEVICE", "cuda"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument("--auto-load-keys", dest="auto_load_keys", action="store_true", default=None)
    key_group.add_argument("--no-auto-load-keys", dest="auto_load_keys", action="store_false")
    parser.add_argument("--rotation-random-mode", choices=("fresh", "reuse_by_shape"), default="fresh")
    parser.add_argument("--rot-key-limb-limit", action="append", default=[], metavar="ROT:LIMBS")
    parser.add_argument(
        "--bootstrap-strategy",
        choices=("double_hoist", "normal_giant", "normal_bsgs"),
        default=os.environ.get("EASYFHE_BOOTSTRAP_STRATEGY", "double_hoist"),
    )
    parser.add_argument(
        "--secret-key-dist",
        choices=("SPARSE_TERNARY", "UNIFORM_TERNARY"),
        default=os.environ.get("EASYFHE_SECRET_KEY_DIST", "SPARSE_TERNARY"),
    )
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--trace-states", action="store_true")
    return parser.parse_args()


def _sync(ctx):
    if ctx.device == "cuda":
        torch.cuda.synchronize()


def _build_runtime(args):
    args.total = max(1, args.warmup + args.iters)
    config = build_config(args)
    bootstrap_extra_depth = bs.depth(
        log_bs_slots=config.log_bs_slots,
        level_budget=config.level_budgets,
        secret_key_dist=config.secret_key_dist,
    )
    bootstrap_rotations = bs.plan_rot_keys(
        log_n=config.log_n,
        log_bs_slots=config.log_bs_slots,
        level_budget=config.level_budgets,
        secret_key_dist=config.secret_key_dist,
    )
    rotations = tuple(dict.fromkeys([*config.rotate_indices, *bootstrap_rotations]))
    client, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=config.max_levels_remaining + bootstrap_extra_depth,
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
    bootstrap_material = {}
    for log_bs_slots, level_budget in zip(config.log_bs_slots, config.level_budgets):
        constants, plan = bs.generate(
            ctx,
            log_bs_slots=log_bs_slots,
            level_budget=level_budget,
            max_levels_remaining=config.max_levels_remaining,
            strategy=config.bootstrap_strategy,
        )
        bootstrap_material[int(log_bs_slots)] = (constants, plan)
    weights = WeightPack.from_npz(config.weights_path, cache_mode=config.weight_cache_mode)
    return AespaRuntime(ctx, client, weights, config, bootstrap_material)


@contextmanager
def _profile_wrappers(rt, records, block_records, op_records, state):
    original_bootstrap = model.bs.bootstrap
    original_same = model._same_shape_residual_block
    original_down = model._downsample_residual_block
    originals = {}

    def timed_bootstrap(cipher, crypto_context, constants, plan, *, L0):
        _sync(rt.ctx)
        start = time.perf_counter()
        result = original_bootstrap(cipher, crypto_context, constants, plan, L0=L0)
        _sync(rt.ctx)
        records.append(
            {
                "stage": state.get("stage"),
                "block": state.get("block"),
                "L0": int(L0),
                "in_limbs": int(cipher.state.cur_limbs),
                "out_limbs": int(result.state.cur_limbs),
                "seconds": time.perf_counter() - start,
            }
        )
        return result

    def timed_same(input, spec, rt_arg):
        previous = state.get("block")
        state["block"] = f"block{spec.block_id}"
        try:
            _sync(rt.ctx)
            start = time.perf_counter()
            result = original_same(input, spec, rt_arg)
            _sync(rt.ctx)
            block_records.append(
                {
                    "stage": state.get("stage"),
                    "block": state.get("block"),
                    "kind": "same",
                    "seconds": time.perf_counter() - start,
                }
            )
            return result
        finally:
            state["block"] = previous

    def timed_down(input, spec, rt_arg):
        previous = state.get("block")
        state["block"] = f"block{spec.block_id}"
        try:
            _sync(rt.ctx)
            start = time.perf_counter()
            result = original_down(input, spec, rt_arg)
            _sync(rt.ctx)
            block_records.append(
                {
                    "stage": state.get("stage"),
                    "block": state.get("block"),
                    "kind": "downsample",
                    "seconds": time.perf_counter() - start,
                }
            )
            return result
        finally:
            state["block"] = previous

    def wrap_op(name):
        original = getattr(model, name)
        originals[name] = original

        def timed(*args, **kwargs):
            _sync(rt.ctx)
            start = time.perf_counter()
            result = original(*args, **kwargs)
            _sync(rt.ctx)
            op_records.append(
                {
                    "stage": state.get("stage"),
                    "block": state.get("block"),
                    "op": name,
                    "seconds": time.perf_counter() - start,
                }
            )
            return result

        setattr(model, name, timed)

    model.bs.bootstrap = timed_bootstrap
    model._same_shape_residual_block = timed_same
    model._downsample_residual_block = timed_down
    for name in (
        "initial_conv3x3",
        "conv3x3",
        "conv3x3_sx",
        "pointwise_conv",
        "pointwise_conv_sx",
        "aespa_nonlinear",
        "aespa_add_shortcut",
        "downsample1024to256",
        "downsample256to64",
        "sum_adjacent_slots",
        "broadcast_slot_sum",
        "sum_channel_groups",
    ):
        wrap_op(name)
    try:
        yield
    finally:
        model.bs.bootstrap = original_bootstrap
        model._same_shape_residual_block = original_same
        model._downsample_residual_block = original_down
        for name, original in originals.items():
            setattr(model, name, original)


def _time_stage(name, state, rt, fn, *args):
    state["stage"] = name
    _sync(rt.ctx)
    start = time.perf_counter()
    result = fn(*args)
    _sync(rt.ctx)
    state["stage"] = None
    return result, time.perf_counter() - start


def _infer_profile(image_vector, rt, bootstrap_records, block_records, op_records):
    state = {"stage": None, "block": None}
    timings = {}
    with _profile_wrappers(rt, bootstrap_records, block_records, op_records, state):
        input_cipher, timings["encrypt"] = _time_stage(
            "encrypt",
            state,
            rt,
            model.encrypt_input,
            image_vector,
            rt,
        )
        first_layer, timings["initial"] = _time_stage("initial", state, rt, model.initial_layer, input_cipher, rt)
        res_layer1, timings["layer1"] = _time_stage("layer1", state, rt, model.layer1, first_layer, rt)
        res_layer2, timings["layer2"] = _time_stage("layer2", state, rt, model.layer2, res_layer1, rt)
        res_layer3, timings["layer3"] = _time_stage("layer3", state, rt, model.layer3, res_layer2, rt)
        final_res, timings["final"] = _time_stage("final", state, rt, model.final_layer, res_layer3, rt)
    timings["model"] = sum(timings[name] for name in ("initial", "layer1", "layer2", "layer3", "final"))
    timings["end_to_end"] = timings["encrypt"] + timings["model"]
    return final_res, timings


def _print_state_trace(trace):
    if not trace:
        return
    print("\nCipher states before conv/downsample ops:")
    for idx, row in enumerate(trace, 1):
        label = row["op"]
        if "kernel_group" in row:
            label += f" {row['kernel_group']}"
        elif "bias_key" in row:
            label += f" {row['bias_key']}"
        print(
            f"{idx:02d}. {label:48s} "
            f"limbs={row['cur_limbs']:2d} noise={row['noise_deg']} slots={row['slots']:5d}"
        )


def _sum_rows(rows, key):
    totals = {}
    for row in rows:
        totals[row[key]] = totals.get(row[key], 0.0) + row["seconds"]
    return totals


def _print_op_summary(op_records):
    if not op_records:
        return
    by_op = _sum_rows(op_records, "op")
    by_stage_op = {}
    for row in op_records:
        key = (row["stage"], row["op"])
        by_stage_op[key] = by_stage_op.get(key, 0.0) + row["seconds"]

    print("\nOps by type:")
    for op, seconds in sorted(by_op.items(), key=lambda item: item[1], reverse=True):
        print(f"{op:22s} total={_format_seconds(seconds)}")

    print("\nOps by stage:")
    for (stage, op), seconds in sorted(by_stage_op.items(), key=lambda item: (str(item[0][0]), -item[1])):
        print(f"{stage}.{op:22s} {_format_seconds(seconds)}")


def _print_cache(weights):
    info = weights.cache_info()
    print(
        "weight cache:",
        f"mode={info['mode']}",
        f"middle={info['middle_entries']}({_format_bytes(info['middle_bytes'])})",
        f"plain={info['plain_entries']}({_format_bytes(info['plain_bytes'])})",
        f"scalar={info.get('scalar_entries', 0)}({_format_bytes(info.get('scalar_bytes', 0))})",
        f"plain_hits={info['plain_hits']}",
        f"plain_misses={info['plain_misses']}",
        f"scalar_hits={info.get('scalar_hits', 0)}",
        f"scalar_misses={info.get('scalar_misses', 0)}",
        f"middle_hits={info['middle_hits']}",
        f"middle_misses={info['middle_misses']}",
    )


def main():
    args = _parse_args()
    rt = _build_runtime(args)
    bootstrap_plan = next(iter(rt.bootstrap_material.values()))[1]
    print("================ ResNet20 AESPA inference benchmark ================")
    print(f"device: {rt.ctx.device}")
    print(f"bootstrap_strategy: {bootstrap_plan.strategy}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"auto_load_keys: {rt.ctx.auto_load_keys_resolved}")

    measured_timings = []
    measured_bootstraps = []
    measured_blocks = []
    measured_ops = []
    for idx in range(args.warmup + args.iters):
        image_vector, label, image_index = read_image(idx, data_dir=rt.config.data_dir)
        bootstrap_records = []
        block_records = []
        op_records = []
        if args.trace_states:
            rt.ctx.aespa_state_trace = []
        _, timings = _infer_profile(image_vector, rt, bootstrap_records, block_records, op_records)
        state_trace = getattr(rt.ctx, "aespa_state_trace", None)
        if args.trace_states:
            rt.ctx.aespa_state_trace = None
        is_warmup = idx < args.warmup
        tag = "warmup" if is_warmup else "measure"
        print(
            f"[{tag} {idx + 1}/{args.warmup + args.iters}]",
            f"index={image_index}",
            f"label={label}",
            f"model={_format_seconds(timings['model'])}",
            f"end_to_end={_format_seconds(timings['end_to_end'])}",
        )
        print(
            "    layers:",
            " ".join(
                f"{name}={_format_seconds(timings[name])}"
                for name in ("encrypt", "initial", "layer1", "layer2", "layer3", "final")
            ),
        )
        print(
            "    bootstrap:",
            f"count={len(bootstrap_records)}",
            f"total={_format_seconds(sum(row['seconds'] for row in bootstrap_records))}",
            " ".join(
                f"{row['stage']}.{row['block']}={_format_seconds(row['seconds'])}"
                for row in bootstrap_records
            ),
        )
        print(
            "    blocks:",
            " ".join(
                f"{row['stage']}.{row['block']}={_format_seconds(row['seconds'])}"
                for row in block_records
            ),
        )
        print(
            "    ops:",
            " ".join(
                f"{op}={_format_seconds(seconds)}"
                for op, seconds in sorted(_sum_rows(op_records, "op").items(), key=lambda item: item[1], reverse=True)
            ),
        )
        if args.trace_states and not is_warmup:
            _print_state_trace(state_trace)
        if not is_warmup:
            measured_timings.append(timings)
            measured_bootstraps.extend(bootstrap_records)
            measured_blocks.extend(block_records)
            measured_ops.extend(op_records)

    if measured_timings:
        print("\n================ measured summary ================")
        for name in ("encrypt", "initial", "layer1", "layer2", "layer3", "final", "model", "end_to_end"):
            avg = sum(row[name] for row in measured_timings) / len(measured_timings)
            print(f"{name:10s} avg={_format_seconds(avg)}")

    if measured_bootstraps:
        print("\nBootstrap calls:")
        for idx, row in enumerate(measured_bootstraps, 1):
            print(
                f"{idx:2d}. {row['stage']}.{row['block']}",
                f"time={_format_seconds(row['seconds'])}",
                f"L0={row['L0']}",
                f"in={row['in_limbs']}",
                f"out={row['out_limbs']}",
            )
        by_stage = _sum_rows(measured_bootstraps, "stage")
        print("Bootstrap by stage:", " ".join(f"{k}={_format_seconds(v)}" for k, v in by_stage.items()))
        print(f"Bootstrap total: {_format_seconds(sum(row['seconds'] for row in measured_bootstraps))}")

    if measured_blocks:
        print("\nBlocks:")
        for row in measured_blocks:
            print(
                f"{row['stage']}.{row['block']}",
                f"kind={row['kind']}",
                f"time={_format_seconds(row['seconds'])}",
            )

    _print_op_summary(measured_ops)

    _print_cache(rt.weights)


if __name__ == "__main__":
    main()
