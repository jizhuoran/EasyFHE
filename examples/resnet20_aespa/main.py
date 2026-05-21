import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from termcolor import colored
import easyfhe as torch
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe
try:
    from .data import DEFAULT_DATA_DIR, read_image, resolve_test_batch_path
    from .fhe_state import runtime_options_from_args
    from .model import AespaRuntime, encrypt_input, infer_encrypted
    from .weight_pack import WeightPack
except ImportError:
    from data import DEFAULT_DATA_DIR, read_image, resolve_test_batch_path
    from fhe_state import runtime_options_from_args
    from model import AespaRuntime, encrypt_input, infer_encrypted
    from weight_pack import WeightPack

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = os.environ.get("EASYFHE_RESNET20_AESPA_DATA_DIR", str(DEFAULT_DATA_DIR))
WEIGHTS_PATH = os.environ.get(
    "EASYFHE_RESNET20_AESPA_WEIGHTS",
    str(SCRIPT_DIR / "resnet20_aespa_weights.npz"),
)


@dataclass(frozen=True)
class AespaConfig:
    total: int
    data_dir: str
    weights_path: str
    rotate_indices: tuple[int, ...]
    max_levels_remaining: int
    log_bs_slots: tuple[int, ...]
    log_n: int
    dnum: int
    dcrt_bits: int
    first_mod: int
    level_budgets: tuple[tuple[int, int], ...]
    bootstrap_strategy: str
    secret_key_dist: str
    scale_mode: str
    rescale_policy: str
    device: str
    weight_cache_mode: str


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("EASYFHE_DEVICE", "cuda"))
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument("--auto-load-keys", dest="auto_load_keys", action="store_true", default=None)
    key_group.add_argument("--no-auto-load-keys", dest="auto_load_keys", action="store_false")
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
    parser.add_argument("--save-middle", action="store_true")
    parser.add_argument("--save-end", action="store_true")
    parser.add_argument("--total", type=int, default=int(os.environ.get("EASYFHE_TOTAL", "1")))
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
    return parser.parse_known_args()[0]


def build_config(args):
    secret_key_dist = str(
        getattr(args, "secret_key_dist", os.environ.get("EASYFHE_SECRET_KEY_DIST", "SPARSE_TERNARY"))
    ).upper()
    return AespaConfig(
        total=args.total,
        data_dir=DATA_DIR,
        weights_path=WEIGHTS_PATH,
        rotate_indices=(
            -8192, -4096, -1024, -768, -256, -192, -64, -33, -32, -31, -17, -16,
            -15, -9, -8, -7, -1, 1, 2, 4, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33,
            48, 64, 128, 256, 512, 1024, 2048, 12288, 24576,
        ),
        max_levels_remaining=12,
        log_bs_slots=(14,),
        log_n=16,
        dnum=int(os.environ.get("EASYFHE_DNUM", "3")),
        dcrt_bits=int(os.environ.get("EASYFHE_DCRT_BITS", "59")),
        first_mod=int(os.environ.get("EASYFHE_FIRST_MOD", "60")),
        level_budgets=((4, 4),),
        bootstrap_strategy=getattr(
            args,
            "bootstrap_strategy",
            os.environ.get("EASYFHE_BOOTSTRAP_STRATEGY", "double_hoist"),
        ),
        secret_key_dist=secret_key_dist,
        scale_mode="fixed",
        rescale_policy="manual",
        device=args.device,
        weight_cache_mode=os.environ.get("EASYFHE_WEIGHT_CACHE_MODE", "plain"),
    )


def _print_config(config):
    print("rotate_index_list: ", list(config.rotate_indices))
    print("maxLevelsRemaining: ", config.max_levels_remaining)
    print("logBsSlots_list: ", list(config.log_bs_slots))
    print("logN: ", config.log_n)
    print("dnum: ", config.dnum)
    print("dcrtBits: ", config.dcrt_bits)
    print("firstMod: ", config.first_mod)
    print("levelBudget_list: ", [list(level_budget) for level_budget in config.level_budgets])
    print("bootstrapStrategy: ", config.bootstrap_strategy)
    print("secretKeyDist: ", config.secret_key_dist)
    print("scaleMode: ", config.scale_mode)
    print("rescalePolicy: ", config.rescale_policy)
    print("weightCacheMode: ", config.weight_cache_mode)
    print("\n\n")
    print("device: ", config.device)
    print("data_dir=", config.data_dir)
    print("test_batch=", resolve_test_batch_path(config.data_dir))
    print("weights_path=", config.weights_path)


def _decrypt_prediction(final_res, rt):
    try:
        clear_result = rt.client.decrypt(final_res).cpu().numpy().reshape(-1)[:10]
        return clear_result, np.argmax(clear_result)
    except RuntimeError as e:
        print(f"Decryption failed: {e}")
        return None, 11


def _sync_device(rt):
    if rt.ctx.device == "cuda":
        torch.cuda.synchronize()


def _format_seconds(seconds):
    return f"{seconds:.3f}s"


def _format_accuracy(correct, total):
    percent = 100.0 * correct / total if total else 0.0
    return f"{correct}/{total} ({percent:.2f}%)"


def _format_bytes(num_bytes):
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0


def _print_time_series(label, times):
    if not times:
        return
    avg = sum(times) / len(times)
    print(
        f"{label}:",
        f"avg={_format_seconds(avg)}",
        f"min={_format_seconds(min(times))}",
        f"max={_format_seconds(max(times))}",
    )
    if len(times) > 1:
        warm_avg = sum(times[1:]) / (len(times) - 1)
        print(f"{label} excluding first image: avg={_format_seconds(warm_avg)}")


def _print_timing_summary(encrypt_times, infer_times, total_seconds, correct, total):
    print("\n================ dataset summary ================")
    print(f"accuracy: {_format_accuracy(correct, total)}")
    print(f"wall time: {_format_seconds(total_seconds)}")
    _print_time_series("encrypt time", encrypt_times)
    _print_time_series("inference time", infer_times)


def _print_weight_cache_summary(weights):
    if not hasattr(weights, "cache_info"):
        return
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


def run_dataset(rt):
    total = rt.config.total
    encrypt_times = []
    infer_times = []
    correct = 0
    dataset_start = time.perf_counter()

    print("\n================ run dataset ================")
    print(f"images: {total}")
    print(f"device: {rt.ctx.device}")

    for i in range(total):
        image_vector, label, index = read_image(i, data_dir=rt.config.data_dir)
        item_start = time.perf_counter()

        _sync_device(rt)
        encrypt_start = time.perf_counter()
        input_cipher = encrypt_input(image_vector, rt)
        _sync_device(rt)
        encrypt_seconds = time.perf_counter() - encrypt_start
        encrypt_times.append(encrypt_seconds)

        infer_start = time.perf_counter()
        final_res = infer_encrypted(input_cipher, rt)
        _sync_device(rt)
        infer_seconds = time.perf_counter() - infer_start
        infer_times.append(infer_seconds)

        decrypt_start = time.perf_counter()
        logits, max_element_idx = _decrypt_prediction(final_res, rt)
        decrypt_seconds = time.perf_counter() - decrypt_start
        item_seconds = time.perf_counter() - item_start

        is_correct = label == max_element_idx
        if is_correct:
            correct += 1
        status = colored("correct", "green") if is_correct else colored("wrong", "red")

        print(
            f"[{i + 1}/{total}] index={index} label={label} "
            f"prediction={max_element_idx} {status}"
        )
        print(
            "    "
            f"encrypt={_format_seconds(encrypt_seconds)} "
            f"infer={_format_seconds(infer_seconds)} "
            f"decrypt={_format_seconds(decrypt_seconds)} "
            f"total={_format_seconds(item_seconds)} "
            f"accuracy={_format_accuracy(correct, i + 1)}"
        )
        if logits is not None:
            print("    logits=", np.array2string(logits, precision=6, separator=", "))

    total_seconds = time.perf_counter() - dataset_start
    _print_timing_summary(encrypt_times, infer_times, total_seconds, correct, total)
    _print_weight_cache_summary(rt.weights)


def resnet20(config=None, args=None):
    if args is None:
        args = _parse_args()
    if config is None:
        config = build_config(args)

    test_batch_path = resolve_test_batch_path(config.data_dir)
    if not test_batch_path.exists():
        raise ValueError(f"CIFAR-10 test batch {test_batch_path} does not exist!")

    _print_config(config)
    options = runtime_options_from_args(args)
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
    client, cryptoContext = fhe.generate_client_context(
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
        ),
        device=config.device,
        options=options,
    )
    bootstrap_material = {}
    for log_bs_slots, level_budget in zip(config.log_bs_slots, config.level_budgets):
        constants, plan = bs.generate(
            cryptoContext,
            log_bs_slots=log_bs_slots,
            level_budget=level_budget,
            max_levels_remaining=config.max_levels_remaining,
            strategy=config.bootstrap_strategy,
        )
        bootstrap_material[int(log_bs_slots)] = (constants, plan)
    print("cryptoContext: ", cryptoContext)
    weights = WeightPack.from_npz(config.weights_path, cache_mode=config.weight_cache_mode)
    print("weights loaded:", len(weights))

    run_dataset(AespaRuntime(cryptoContext, client, weights, config, bootstrap_material))


if __name__ == "__main__":
    resnet20()
