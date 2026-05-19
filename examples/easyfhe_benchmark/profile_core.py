from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

try:
    from .paths import ensure_repo_on_path
except ImportError:
    from paths import ensure_repo_on_path

ensure_repo_on_path()

import easyfhe as torch
import easyfhe.fhe as fhe
import easyfhe.bs.openfhe as bs

from easyfhe_benchmark import common as bench


def sync_if_needed(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_isolated(op_fn: Callable[[], Any], *, warmup_iters: int, timed_iters: int, device: str) -> tuple[Any, List[float]]:
    last_result = None
    for _ in range(int(warmup_iters)):
        last_result = op_fn()
    sync_if_needed(device)

    samples_us: List[float] = []
    for _ in range(int(timed_iters)):
        sync_if_needed(device)
        start = time.perf_counter()
        last_result = op_fn()
        sync_if_needed(device)
        samples_us.append((time.perf_counter() - start) * 1e6)
    return last_result, samples_us


def profile_batched(
    op_fn: Callable[[], Any],
    *,
    warmup_iters: int,
    timed_iters: int,
    batch_size: int,
    device: str,
) -> tuple[Any, List[float]]:
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")

    last_result = None
    for _ in range(int(warmup_iters)):
        last_result = op_fn()
    sync_if_needed(device)

    samples_us: List[float] = []
    for _ in range(int(timed_iters)):
        sync_if_needed(device)
        start = time.perf_counter()
        for _ in range(int(batch_size)):
            last_result = op_fn()
        sync_if_needed(device)
        elapsed_us = (time.perf_counter() - start) * 1e6
        samples_us.append(elapsed_us / float(batch_size))
    return last_result, samples_us


def profile_once(op_fn: Callable[[], Any], *, warmup_iters: int, timed_iters: int, device: str) -> tuple[Any, List[float]]:
    return profile_isolated(op_fn, warmup_iters=warmup_iters, timed_iters=timed_iters, device=device)


def timing_summary(samples_us: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples_us, dtype=np.float64)
    return {
        "median_us": float(np.median(arr)),
        "p90_us": float(np.percentile(arr, 90)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "mean_us": float(np.mean(arr)),
    }


def build_bootstrap_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    log_bs_slots = int(case.extra["logBsSlots"])
    level_budget = [int(x) for x in case.extra["level_budget"]]
    target_limbs = int(case.extra["target_limbs"])
    cipher = bench.make_cipher(openfhe_context, crypto_context, bench.vector_values(case.slots), case.cur_limbs, case.slots)
    bootstrap_material = getattr(crypto_context, "benchmark_bootstrap_constants", {}).get(
        (log_bs_slots, tuple(level_budget))
    )
    if bootstrap_material is None:
        bs_keys, constants, plan = bs.generate(
            crypto_context,
            log_bs_slots=log_bs_slots,
            level_budget=level_budget,
            max_levels_remaining=int(getattr(crypto_context, "maxLevelsRemaining", target_limbs)),
        )
        crypto_context.addkeys(bs_keys)
    else:
        constants, plan = bootstrap_material

    def op():
        return bs.bootstrap(cipher, crypto_context, constants, plan, L0=target_limbs)

    probe = op()
    return op, {
        "logBsSlots": int(log_bs_slots),
        "level_budget": list(level_budget),
        "target_limbs": int(target_limbs),
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }
