from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

try:
    from .paths import ensure_repo_on_path
except ImportError:
    from paths import ensure_repo_on_path

ensure_repo_on_path()

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ops import homo as homo_ops
from easyfhe.fhe.ops import rotation

from easyfhe_benchmark import common as bench
from easyfhe_benchmark import profile_core


PROFILED_OPS = [
    "encode",
    "add",
    "add_pt",
    "mul_pt",
    "mul",
    "rescale",
    "force_rescale",
    "drop_last_elements",
    "modup_to_ext",
    "moddown_from_ext",
    "square",
    "rotate",
    "fast_rotate",
]

CHEAP_OPS = {
    "add",
    "add_pt",
    "force_rescale",
    "drop_last_elements",
    "rotate",
}

MEDIUM_OPS = {
    "encode",
    "mul_pt",
    "mul",
    "rescale",
    "modup_to_ext",
    "moddown_from_ext",
    "square",
    "fast_rotate",
}


def _rescale_one_level(cipher, crypto_context):
    return fhe.align_to(
        cipher,
        fhe.CipherState(cipher.cur_limbs - 1, cipher.noise_deg - 1),
        crypto_context,
    )


def _parse_ops(values: Sequence[str]) -> List[str]:
    ops = bench.parse_ops(values)
    unknown = [op for op in ops if op not in PROFILED_OPS]
    if unknown:
        raise ValueError(f"Unsupported ops for timed profiling: {unknown}")
    return ops


def _profile_case_kind(op: str) -> str:
    if op in CHEAP_OPS:
        return "cheap"
    if op in MEDIUM_OPS:
        return "medium"
    raise ValueError(f"Missing repetition policy for op={op}")


def _repetition_policy(args: argparse.Namespace, op: str) -> tuple[int, int]:
    kind = _profile_case_kind(op)
    if kind == "cheap":
        return int(args.warmup_cheap), int(args.timed_cheap)
    return int(args.warmup_medium), int(args.timed_medium)


def _make_cases(args: argparse.Namespace) -> List[bench.BenchmarkCase]:
    ops = _parse_ops(args.ops)
    cases: List[bench.BenchmarkCase] = []
    for op in ops:
        if op in bench.KEYSWITCH_LIKE_OPS:
            dnum_values = [int(value) for value in args.dnum_values]
        else:
            dnum_values = [int(args.default_dnum)]

        if op == "encode":
            for slots in [int(value) for value in args.encode_slot_values]:
                for cur_limbs in range(int(args.limb_min), int(args.limb_max) + 1):
                    for dnum in dnum_values:
                        cases.append(
                            bench.BenchmarkCase(
                                op=op,
                                cur_limbs=int(cur_limbs),
                                dnum=int(dnum),
                                slots=int(slots),
                                ct_size=1,
                                extra={},
                            )
                        )
        elif op in {"rotate", "fast_rotate"}:
            for step in [int(value) for value in args.rotate_steps]:
                for cur_limbs in range(int(args.limb_min), int(args.limb_max) + 1):
                    for dnum in dnum_values:
                        cases.append(
                            bench.BenchmarkCase(
                                op=op,
                                cur_limbs=int(cur_limbs),
                                dnum=int(dnum),
                                slots=int(args.basic_slots),
                                ct_size=2,
                                extra={"step": int(step)},
                            )
                        )
        else:
            for cur_limbs in range(int(args.limb_min), int(args.limb_max) + 1):
                for dnum in dnum_values:
                    cases.append(
                        bench.BenchmarkCase(
                            op=op,
                            cur_limbs=int(cur_limbs),
                            dnum=int(dnum),
                            slots=int(args.basic_slots),
                            ct_size=2,
                            extra={},
                        )
                    )
    if args.limit_cases is not None:
        return cases[: int(args.limit_cases)]
    return cases


def _make_report(
    args: argparse.Namespace,
    *,
    cases: Sequence[bench.BenchmarkCase],
    results: Sequence[Dict[str, Any]],
    timing_modes: Sequence[str],
) -> Dict[str, Any]:
    completed_ids = {str(result.get("case_id")) for result in results if result.get("case_id")}
    pending = [
        row
        for row in (bench.case_manifest_row(case) for case in cases)
        if row["case_id"] not in completed_ids
    ]
    return {
        "config": {
            "device": args.device,
            "rotation_random_mode": str(args.rotation_random_mode),
            "max_levels_remaining": int(args.max_levels_remaining),
            "logN": int(args.logN),
            "dnum_values": [int(value) for value in args.dnum_values],
            "default_dnum": int(args.default_dnum),
            "limb_min": int(args.limb_min),
            "limb_max": int(args.limb_max),
            "basic_slots": int(args.basic_slots),
            "encode_slot_values": [int(value) for value in args.encode_slot_values],
            "rotate_steps": [int(value) for value in args.rotate_steps],
            "ops": _parse_ops(args.ops),
            "warmup_cheap": int(args.warmup_cheap),
            "timed_cheap": int(args.timed_cheap),
            "warmup_medium": int(args.warmup_medium),
            "timed_medium": int(args.timed_medium),
            "timing_modes": list(timing_modes),
            "batched_group_size": int(args.batched_group_size),
            "resume": not bool(args.no_resume),
            "rerun_failed": bool(args.rerun_failed),
        },
        "case_manifest": [bench.case_manifest_row(case) for case in cases],
        "resume": {
            "total_cases": len(cases),
            "completed_cases": len(completed_ids),
            "pending_cases": len(pending),
            "pending_case_ids": [row["case_id"] for row in pending],
        },
        "summary": bench.summarize_results(results),
        "results": list(results),
    }


def _write_report(
    output_json: Path,
    args: argparse.Namespace,
    *,
    cases: Sequence[bench.BenchmarkCase],
    results: Sequence[Dict[str, Any]],
    timing_modes: Sequence[str],
) -> None:
    report = _make_report(args, cases=cases, results=results, timing_modes=timing_modes)
    output_json.write_text(json.dumps(report, indent=2))


def _make_mul_pt_input(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any):
    lhs_vals = bench.vector_values(case.slots)
    pt_vals = [0.5 + 0.05 * (index % 3) for index in range(case.slots)]
    lhs = bench.make_cipher(openfhe_context, crypto_context, lhs_vals, case.cur_limbs, case.slots)
    plain = bench.make_plain_from_middle(
        crypto_context,
        pt_vals,
        case.cur_limbs,
        case.slots,
        f"profile_mul_pt_input_{case.cur_limbs}_{case.slots}",
    )
    return lhs, plain


def _build_encode_target(case: bench.BenchmarkCase, crypto_context: Any, _openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    values = bench.vector_values(case.slots)
    middle_value = homo_ops.prepare_plaintext(np.asarray(values, dtype=np.float64), case.slots, crypto_context.N)
    level = bench.level_for_cur_limbs(crypto_context, case.cur_limbs)
    def op():
        return homo_ops.encode(
            middle_value,
            crypto_context,
            level=level,
            slots=case.slots,
            is_ext=False,
        )[1]

    probe = op()
    return op, {
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
        "profile_contract": "middle_to_encode",
    }


def _build_add_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    lhs_vals = bench.vector_values(case.slots)
    rhs_vals = [value * 0.5 for value in lhs_vals]
    lhs = bench.make_cipher(openfhe_context, crypto_context, lhs_vals, case.cur_limbs, case.slots)
    rhs = bench.make_cipher(openfhe_context, crypto_context, rhs_vals, case.cur_limbs, case.slots)

    def op():
        return fhe.homo_add(lhs, rhs, crypto_context)

    probe = op()
    return op, {
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_add_pt_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    lhs_vals = bench.vector_values(case.slots)
    pt_vals = [value * 0.25 for value in lhs_vals]
    lhs = bench.make_cipher(openfhe_context, crypto_context, lhs_vals, case.cur_limbs, case.slots)
    plain = bench.make_plain_from_middle(
        crypto_context,
        pt_vals,
        case.cur_limbs,
        case.slots,
        f"profile_add_pt_{case.cur_limbs}_{case.slots}",
    )

    def op():
        return fhe.homo_add_pt(lhs, plain, crypto_context)

    probe = op()
    return op, {
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_mul_pt_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    lhs, plain = _make_mul_pt_input(case, crypto_context, openfhe_context)

    def op():
        return fhe.homo_mul_pt(lhs, plain, crypto_context)

    probe = op()
    return op, {
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_mul_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    lhs_vals = bench.vector_values(case.slots)
    rhs_vals = [0.5 + 0.125 * ((index % 6) - 2) for index in range(case.slots)]
    lhs = bench.make_cipher(openfhe_context, crypto_context, lhs_vals, case.cur_limbs, case.slots)
    rhs = bench.make_cipher(openfhe_context, crypto_context, rhs_vals, case.cur_limbs, case.slots)

    def op():
        return fhe.homo_mul(lhs, rhs, crypto_context)

    probe = op()
    return op, {
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_rescale_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    lhs, plain = _make_mul_pt_input(case, crypto_context, openfhe_context)
    mul_result = fhe.homo_mul_pt(lhs, plain, crypto_context)

    def op():
        return _rescale_one_level(mul_result, crypto_context)

    probe = op()
    return op, {
        "input_noise_deg": int(mul_result.noise_deg),
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_force_rescale_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    lhs, plain = _make_mul_pt_input(case, crypto_context, openfhe_context)
    mul_result = fhe.homo_mul_pt(lhs, plain, crypto_context)

    def op():
        return _rescale_one_level(mul_result, crypto_context)

    probe = op()
    return op, {
        "input_noise_deg": int(mul_result.noise_deg),
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_drop_last_elements_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    cipher = bench.make_cipher(openfhe_context, crypto_context, bench.vector_values(case.slots), case.cur_limbs, case.slots)

    def op():
        return fhe.align_to(
            cipher,
            fhe.CipherState(cipher.cur_limbs - 1, cipher.noise_deg),
            crypto_context,
        )

    probe = op()
    return op, {
        "drop_count": 1,
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_modup_to_ext_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    cipher = bench.make_cipher(openfhe_context, crypto_context, bench.vector_values(case.slots), case.cur_limbs, case.slots)

    def op():
        return rotation._modup_to_ext(cipher, crypto_context)

    probe = op()
    return op, {
        "result_cur_limbs": int(probe.cur_limbs),
        "result_is_ext": bool(probe.is_ext),
    }


def _build_moddown_from_ext_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    cipher = bench.make_cipher(openfhe_context, crypto_context, bench.vector_values(case.slots), case.cur_limbs, case.slots)
    ext_cipher = cipher.cipher_like(
        [rotation._scale_to_P_ext(cv, cipher, crypto_context) for cv in cipher.cv],
        is_ext=True,
        cipher_id="assign",
    )

    def op():
        return rotation.moddown_from_ext(ext_cipher, crypto_context)

    probe = op()
    return op, {
        "input_contract": "scale_to_P_ext_then_moddown_from_ext",
        "ext_cur_limbs": int(ext_cipher.cur_limbs),
        "ext_is_ext": bool(ext_cipher.is_ext),
        "result_cur_limbs": int(probe.cur_limbs),
        "result_is_ext": bool(probe.is_ext),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_square_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    cipher = bench.make_cipher(openfhe_context, crypto_context, bench.vector_values(case.slots), case.cur_limbs, case.slots)

    def op():
        return fhe.homo_square(cipher, crypto_context)

    probe = op()
    return op, {
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_rotate_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    step = int(case.extra["step"])
    cipher = bench.make_cipher(openfhe_context, crypto_context, bench.vector_values(case.slots), case.cur_limbs, case.slots)

    def op():
        return fhe.homo_rotate(cipher, step, crypto_context)

    probe = op()
    return op, {
        "step": int(step),
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


def _build_fast_rotate_target(case: bench.BenchmarkCase, crypto_context: Any, openfhe_context: Any) -> tuple[Callable[[], Any], Dict[str, Any]]:
    step = int(case.extra["step"])
    cipher = bench.make_cipher(openfhe_context, crypto_context, bench.vector_values(case.slots), case.cur_limbs, case.slots)

    def op():
        return fhe.fast_rotate(cipher, [step], crypto_context)[0]

    probe = op()
    return op, {
        "step": int(step),
        "result_cur_limbs": int(probe.cur_limbs),
        "result_noise_deg": int(probe.noise_deg),
    }


TARGET_BUILDERS: Dict[str, Callable[[bench.BenchmarkCase, Any, Any], tuple[Callable[[], Any], Dict[str, Any]]]] = {
    "encode": _build_encode_target,
    "add": _build_add_target,
    "add_pt": _build_add_pt_target,
    "mul_pt": _build_mul_pt_target,
    "mul": _build_mul_target,
    "rescale": _build_rescale_target,
    "force_rescale": _build_force_rescale_target,
    "drop_last_elements": _build_drop_last_elements_target,
    "modup_to_ext": _build_modup_to_ext_target,
    "moddown_from_ext": _build_moddown_from_ext_target,
    "square": _build_square_target,
    "rotate": _build_rotate_target,
    "fast_rotate": _build_fast_rotate_target,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Timed profiling harness for foundational EasyFHE ops")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--context-cache-mode", choices=sorted(bench.CONTEXT_CACHE_MODES), default="persistent")
    ap.add_argument("--context-cache-root")
    ap.add_argument("--ops", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rotation-random-mode", choices=["fresh", "reuse_by_shape"], default="reuse_by_shape")
    ap.add_argument("--max-levels-remaining", type=int, default=25)
    ap.add_argument("--logN", type=int, default=16)
    ap.add_argument("--dnum-values", nargs="+", type=int, default=[1, 3, 5, 7])
    ap.add_argument("--default-dnum", type=int, default=3)
    ap.add_argument("--dcrt-bits", type=int, default=52)
    ap.add_argument("--first-mod", type=int, default=55)
    ap.add_argument("--secret-key-dist", default="SPARSE_TERNARY")
    ap.add_argument("--rescale-tech", default="FIXEDMANUAL")
    ap.add_argument("--limb-min", type=int, default=1)
    ap.add_argument("--limb-max", type=int, default=30)
    ap.add_argument("--basic-slots", type=int, default=4096)
    ap.add_argument("--encode-slot-values", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    ap.add_argument("--rotate-steps", nargs="+", type=int, default=[-1])
    ap.add_argument("--warmup-cheap", type=int, default=30)
    ap.add_argument("--timed-cheap", type=int, default=100)
    ap.add_argument("--warmup-medium", type=int, default=20)
    ap.add_argument("--timed-medium", type=int, default=50)
    ap.add_argument("--timing-modes", nargs="+", choices=["isolated", "batched"], default=["isolated"])
    ap.add_argument("--batched-group-size", type=int, default=30)
    ap.add_argument("--include-samples", action="store_true")
    ap.add_argument("--limit-cases", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--rerun-failed", action="store_true")
    args = ap.parse_args()

    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cache = bench.ContextCache(args)
    timing_modes = list(dict.fromkeys(str(mode) for mode in args.timing_modes))
    cases = _make_cases(args)
    valid_case_ids = {bench.case_id(case) for case in cases}
    completed = {} if args.no_resume else bench.load_completed_results(output_json, rerun_failed=bool(args.rerun_failed))
    results: List[Dict[str, Any]] = []
    completed_ids = set()

    for case in cases:
        case_id = bench.case_id(case)
        previous = completed.get(case_id)
        if previous is not None and case_id in valid_case_ids:
            previous = dict(previous)
            previous["case_id"] = case_id
            previous["resumed"] = True
            results.append(previous)
            completed_ids.add(case_id)

    _write_report(output_json, args, cases=cases, results=results, timing_modes=timing_modes)

    for case in cases:
        case_id = bench.case_id(case)
        if case_id in completed_ids:
            continue
        try:
            context_kwargs = bench.case_context_kwargs(case)
            crypto_context, openfhe_context = cache.get(case.dnum, **context_kwargs)
            support_reason = bench.validate_case_support(case)
            if support_reason is not None:
                result = bench.mark_unsupported(case, support_reason)
            elif case.cur_limbs > int(crypto_context.L):
                result = bench.mark_unsupported(case, f"cur_limbs={case.cur_limbs} exceeds context L={crypto_context.L}")
            else:
                op_fn, details = TARGET_BUILDERS[case.op](case, crypto_context, openfhe_context)
                warmup_iters, timed_iters = _repetition_policy(args, case.op)
                timing: Dict[str, Any] = {
                    "warmup_iters": int(warmup_iters),
                    "timed_iters": int(timed_iters),
                    "timing_modes": list(timing_modes),
                }
                if "isolated" in timing_modes:
                    _last_result, isolated_samples_us = profile_core.profile_isolated(
                        op_fn,
                        warmup_iters=warmup_iters,
                        timed_iters=timed_iters,
                        device=str(args.device),
                    )
                    isolated_summary = profile_core.timing_summary(isolated_samples_us)
                    timing.update(isolated_summary)
                    timing["isolated"] = dict(isolated_summary)
                    if args.include_samples:
                        timing["samples_us"] = [float(value) for value in isolated_samples_us]
                        timing["isolated"]["samples_us"] = [float(value) for value in isolated_samples_us]
                if "batched" in timing_modes:
                    _last_result, batched_samples_us = profile_core.profile_batched(
                        op_fn,
                        warmup_iters=warmup_iters,
                        timed_iters=timed_iters,
                        batch_size=int(args.batched_group_size),
                        device=str(args.device),
                    )
                    batched_summary = profile_core.timing_summary(batched_samples_us)
                    timing["batched"] = {
                        **batched_summary,
                        "group_size": int(args.batched_group_size),
                    }
                    if args.include_samples:
                        timing["batched"]["samples_us"] = [float(value) for value in batched_samples_us]
                result = {
                    "case_id": case_id,
                    "op": case.op,
                    "status": "ok",
                    "cur_limbs": case.cur_limbs,
                    "dnum": case.dnum,
                    "slots": case.slots,
                    "ct_size": case.ct_size,
                    "extra": case.extra,
                    "details": details,
                    "timing": {
                        **timing,
                        "warmup_iters": int(warmup_iters),
                        "timed_iters": int(timed_iters),
                    },
                }
        except Exception as exc:
            result = {
                "case_id": case_id,
                "op": case.op,
                "status": "failed",
                "reason": str(exc),
                "traceback": traceback.format_exc(),
                "cur_limbs": case.cur_limbs,
                "dnum": case.dnum,
                "slots": case.slots,
                "ct_size": case.ct_size,
                "extra": case.extra,
            }
        result["case_id"] = case_id
        results.append(result)
        _write_report(output_json, args, cases=cases, results=results, timing_modes=timing_modes)

    report = _make_report(args, cases=cases, results=results, timing_modes=timing_modes)
    summary = report["summary"]
    print(json.dumps({"output_json": str(output_json), "summary": summary}, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
