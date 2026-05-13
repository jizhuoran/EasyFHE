from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

try:
    from .paths import DATA_ROOT, ensure_repo_on_path, repo_path
except ImportError:
    from paths import DATA_ROOT, ensure_repo_on_path, repo_path

ensure_repo_on_path()

import easyfhe as torch

from easyfhe_benchmark import common as bench
from easyfhe_benchmark import profile_core

DEFAULT_MIN_TARGET_LIMBS_BY_LEVEL_BUDGET = {
    (3, 3): 17,
    (4, 4): 19,
}


def _target_limb_padding(secret_key_dist: str) -> int:
    dist = str(secret_key_dist).upper()
    if dist == "SPARSE_TERNARY":
        return 11
    if dist == "UNIFORM_TERNARY":
        return 14
    raise ValueError(f"Unsupported secret_key_dist for bootstrap target bound: {secret_key_dist}")


def _derived_target_limbs_max(args: argparse.Namespace, level_budget: Sequence[int]) -> int:
    return (
        int(args.context_max_levels_remaining)
        + sum(int(x) for x in level_budget)
        + _target_limb_padding(str(args.secret_key_dist))
    )


def _make_base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        save_dir=str(args.save_dir),
        context_cache_mode=str(args.context_cache_mode),
        context_cache_root=str(args.context_cache_root) if args.context_cache_root else None,
        max_levels_remaining=int(args.context_max_levels_remaining),
        logN=int(args.logN),
        dcrt_bits=int(args.dcrt_bits),
        first_mod=int(args.first_mod),
        secret_key_dist=str(args.secret_key_dist),
        rescale_tech=str(args.rescale_tech),
        device=str(args.device),
        rotation_random_mode=str(args.rotation_random_mode),
    )


def _bucket_name(dnum: int, logbs: int, level_budget: Sequence[int]) -> str:
    slots = 1 << int(logbs)
    return f"d{int(dnum)}_s{int(slots)}_lb{int(level_budget[0])}{int(level_budget[1])}"


def _effective_target_limbs_max(args: argparse.Namespace, level_budget: Sequence[int]) -> int:
    explicit_max = args.target_limbs_max
    if explicit_max is None:
        return _derived_target_limbs_max(args, level_budget)
    return int(explicit_max)


def _target_limb_values(args: argparse.Namespace, level_budget: Sequence[int]) -> List[int]:
    bucket_min = DEFAULT_MIN_TARGET_LIMBS_BY_LEVEL_BUDGET.get(tuple(int(x) for x in level_budget), int(args.target_limbs_min))
    start = max(int(args.target_limbs_min), int(bucket_min))
    stop = _effective_target_limbs_max(args, level_budget)
    if start > stop:
        return []
    return list(range(int(start), int(stop) + 1))


def _bucket_defs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    defs: List[Dict[str, Any]] = []
    for dnum in [int(value) for value in args.dnum_values]:
        for logbs in [int(value) for value in args.logbs_values]:
            for raw_budget in args.level_budgets:
                left, right = str(raw_budget).split(",", 1)
                level_budget = [int(left), int(right)]
                defs.append(
                    {
                        "name": _bucket_name(dnum, logbs, level_budget),
                        "dnum": int(dnum),
                        "logbs": int(logbs),
                        "slots": int(1 << int(logbs)),
                        "level_budget": level_budget,
                    }
                )
    return defs


def _make_case(bucket: Dict[str, Any], target_limbs: int) -> bench.BenchmarkCase:
    return bench.BenchmarkCase(
        op="bootstrap",
        cur_limbs=2,
        dnum=int(bucket["dnum"]),
        slots=int(bucket["slots"]),
        ct_size=2,
        extra={
            "logBsSlots": int(bucket["logbs"]),
            "level_budget": [int(x) for x in bucket["level_budget"]],
            "target_limbs": int(target_limbs),
        },
    )


def _profile_case(
    case: bench.BenchmarkCase,
    *,
    crypto_context: Any,
    openfhe_context: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    support_reason = bench.validate_case_support(case)
    if support_reason is not None:
        return bench.mark_unsupported(case, support_reason)
    if case.cur_limbs > int(crypto_context.L):
        return bench.mark_unsupported(case, f"cur_limbs={case.cur_limbs} exceeds context L={crypto_context.L}")

    op_fn, details = profile_core.build_bootstrap_target(case, crypto_context, openfhe_context)
    _last_result, samples_us = profile_core.profile_once(
        op_fn,
        warmup_iters=int(args.warmup_heavy),
        timed_iters=int(args.timed_heavy),
        device=str(args.device),
    )
    timing = profile_core.timing_summary(samples_us)
    if args.include_samples:
        timing["samples_us"] = [float(value) for value in samples_us]
    return {
        "op": case.op,
        "status": "ok",
        "cur_limbs": int(case.cur_limbs),
        "dnum": int(case.dnum),
        "slots": int(case.slots),
        "ct_size": int(case.ct_size),
        "extra": dict(case.extra),
        "details": details,
        "timing": {
            **timing,
            "warmup_iters": int(args.warmup_heavy),
            "timed_iters": int(args.timed_heavy),
        },
    }


def _bucket_report(
    bucket: Dict[str, Any],
    *,
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
    cases: Sequence[bench.BenchmarkCase],
) -> Dict[str, Any]:
    summary = {"ok": 0, "unsupported": 0, "failed": 0}
    for row in results:
        summary[str(row["status"])] += 1
    completed_ids = {str(row.get("case_id")) for row in results if row.get("case_id")}
    pending = [
        row
        for row in (bench.case_manifest_row(case) for case in cases)
        if row["case_id"] not in completed_ids
    ]
    return {
        "config": {
            "device": str(args.device),
            "context_max_levels_remaining": int(args.context_max_levels_remaining),
            "logN": int(args.logN),
            "dcrt_bits": int(args.dcrt_bits),
            "first_mod": int(args.first_mod),
            "secret_key_dist": str(args.secret_key_dist),
            "rescale_tech": str(args.rescale_tech),
            "warmup_heavy": int(args.warmup_heavy),
            "timed_heavy": int(args.timed_heavy),
            "target_limb_values": _target_limb_values(args, bucket["level_budget"]),
            "bucket": {
                "name": str(bucket["name"]),
                "dnum": int(bucket["dnum"]),
                "logbs": int(bucket["logbs"]),
                "slots": int(bucket["slots"]),
                "level_budget": [int(x) for x in bucket["level_budget"]],
            },
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
        "summary": summary,
        "results": results,
    }


def _write_bucket_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def _load_completed_results(path: Path, *, rerun_failed: bool) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    completed: Dict[str, Dict[str, Any]] = {}
    for result in data.get("results", []):
        case_id = result.get("case_id")
        if not case_id:
            continue
        if bench.result_is_complete(result, rerun_failed=rerun_failed):
            completed[str(case_id)] = result
    return completed


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _update_csv(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any] | None:
    if not args.csv_inout:
        return None
    from easyfhe_benchmark.tools import update_bootstrap_rows_in_csv as update_csv

    csv_in = repo_path(args.csv_inout)
    existing_rows = update_csv._read_csv_rows(csv_in)
    bootstrap_rows = update_csv._bootstrap_rows_from_timed_dir(
        out_dir,
        accelerator=str(args.accelerator),
        allowed_names=None,
    )
    merged_rows = update_csv.merge_rows(existing_rows, bootstrap_rows, replace_bootstrap=True)
    update_csv.write_csv(csv_in, merged_rows)
    return {
        "csv_out": str(csv_in),
        "bootstrap_rows_added_or_updated": len(bootstrap_rows),
        "total_rows": len(merged_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bucketed bootstrap profiling runner with one shared context per (dnum, slots, levelBudget) bucket")
    ap.add_argument("--save-dir", default=str(DATA_ROOT / "context_store"))
    ap.add_argument("--output-dir", default=str(DATA_ROOT / "bootstrap_bucketed_profiling"))
    ap.add_argument("--context-cache-mode", choices=sorted(bench.CONTEXT_CACHE_MODES), default="persistent")
    ap.add_argument("--context-cache-root")
    ap.add_argument("--csv-inout")
    ap.add_argument("--accelerator", default="sim")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rotation-random-mode", choices=["fresh", "reuse_by_shape"], default="reuse_by_shape")
    ap.add_argument("--context-max-levels-remaining", type=int, default=25)
    ap.add_argument("--target-limbs-min", type=int, default=1)
    ap.add_argument("--target-limbs-max", type=int)
    ap.add_argument("--dnum-values", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--logbs-values", nargs="+", type=int, default=[12, 13, 14])
    ap.add_argument("--level-budgets", nargs="+", default=["3,3", "4,4"])
    ap.add_argument("--warmup-heavy", type=int, default=1)
    ap.add_argument("--timed-heavy", type=int, default=3)
    ap.add_argument("--logN", type=int, default=16)
    ap.add_argument("--dcrt-bits", type=int, default=52)
    ap.add_argument("--first-mod", type=int, default=55)
    ap.add_argument("--secret-key-dist", default="SPARSE_TERNARY")
    ap.add_argument("--rescale-tech", default="FIXEDMANUAL")
    ap.add_argument("--limit-buckets", type=int)
    ap.add_argument("--include-samples", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--rerun-failed", action="store_true")
    args = ap.parse_args(argv)

    max_bounds = [_effective_target_limbs_max(args, raw.split(",")) for raw in args.level_budgets]
    if int(args.target_limbs_min) > min(int(v) for v in max_bounds):
        raise ValueError("--target-limbs-min must be <= --target-limbs-max")

    out_dir = repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.context_cache_mode == "transient" and not args.context_cache_root:
        args.context_cache_root = str(out_dir / ".context_cache")

    bucket_defs = _bucket_defs(args)
    if args.limit_buckets is not None:
        bucket_defs = bucket_defs[: int(args.limit_buckets)]

    manifest_rows: List[Dict[str, Any]] = []
    for bucket in bucket_defs:
        for target_limbs in _target_limb_values(args, bucket["level_budget"]):
            case = _make_case(bucket, int(target_limbs))
            manifest_rows.append(
                {
                    **bench.case_manifest_row(case),
                    "bucket": {
                        "name": str(bucket["name"]),
                        "dnum": int(bucket["dnum"]),
                        "logbs": int(bucket["logbs"]),
                        "slots": int(bucket["slots"]),
                        "level_budget": [int(x) for x in bucket["level_budget"]],
                    },
                }
            )
    (out_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "device": str(args.device),
                    "rotation_random_mode": str(args.rotation_random_mode),
                    "context_max_levels_remaining": int(args.context_max_levels_remaining),
                    "target_limbs_min": int(args.target_limbs_min),
                    "target_limbs_max": {
                        str(raw): int(_effective_target_limbs_max(args, raw.split(","))) for raw in args.level_budgets
                    },
                    "dnum_values": [int(x) for x in args.dnum_values],
                    "logbs_values": [int(x) for x in args.logbs_values],
                    "level_budgets": [str(x) for x in args.level_budgets],
                    "bucket_count": len(bucket_defs),
                    "case_count": len(manifest_rows),
                },
                "cases": manifest_rows,
            },
            indent=2,
        )
    )

    run_rows: List[Dict[str, Any]] = []
    for bucket in bucket_defs:
        output_json = out_dir / f"timed__{bucket['name']}.json"
        target_limb_values = _target_limb_values(args, bucket["level_budget"])
        cases = [_make_case(bucket, int(target_limbs)) for target_limbs in target_limb_values]
        valid_case_ids = {bench.case_id(case) for case in cases}
        results: List[Dict[str, Any]] = []
        completed_ids = set()
        if not args.no_resume:
            for case_id, previous in _load_completed_results(output_json, rerun_failed=bool(args.rerun_failed)).items():
                if case_id not in valid_case_ids:
                    continue
                previous = dict(previous)
                previous["case_id"] = str(case_id)
                previous["resumed"] = True
                results.append(previous)
                completed_ids.add(str(case_id))
        _write_bucket_json(output_json, _bucket_report(bucket, args=args, results=results, cases=cases))
        try:
            pending_cases = [case for case in cases if bench.case_id(case) not in completed_ids]
            if not pending_cases:
                report = _bucket_report(bucket, args=args, results=results, cases=cases)
                run_rows.append(
                    {
                        "bucket": {
                            "name": str(bucket["name"]),
                            "dnum": int(bucket["dnum"]),
                            "slots": int(bucket["slots"]),
                            "level_budget": [int(x) for x in bucket["level_budget"]],
                        },
                        "status": "ok" if int(report["summary"]["failed"]) == 0 else "failed",
                        "summary": dict(report["summary"]),
                        "timed_output_json": str(output_json),
                        "resumed": True,
                    }
                )
                continue

            base_args = _make_base_args(args)
            cache = bench.ContextCache(base_args)
            crypto_context, openfhe_context = cache.get(
                int(bucket["dnum"]),
                need_bootstrap=True,
                app_rot_indices=[],
                log_bs_slots_list=[int(bucket["logbs"])],
                level_budget_list=[[int(x) for x in bucket["level_budget"]]],
            )
            for case in pending_cases:
                case_id = bench.case_id(case)
                try:
                    result = _profile_case(
                        case,
                        crypto_context=crypto_context,
                        openfhe_context=openfhe_context,
                        args=args,
                    )
                except Exception as exc:
                    result = {
                        "case_id": case_id,
                        "op": case.op,
                        "status": "failed",
                        "reason": str(exc),
                        "traceback": traceback.format_exc(),
                        "cur_limbs": int(case.cur_limbs),
                        "dnum": int(case.dnum),
                        "slots": int(case.slots),
                        "ct_size": int(case.ct_size),
                        "extra": dict(case.extra),
                    }
                result["case_id"] = case_id
                results.append(result)
                _write_bucket_json(output_json, _bucket_report(bucket, args=args, results=results, cases=cases))
            report = _bucket_report(bucket, args=args, results=results, cases=cases)
            _write_bucket_json(output_json, report)
            run_rows.append(
                {
                    "bucket": {
                        "name": str(bucket["name"]),
                        "dnum": int(bucket["dnum"]),
                        "slots": int(bucket["slots"]),
                        "level_budget": [int(x) for x in bucket["level_budget"]],
                    },
                    "status": "ok" if int(report["summary"]["failed"]) == 0 else "failed",
                    "summary": dict(report["summary"]),
                    "timed_output_json": str(output_json),
                }
            )
            del crypto_context
            del openfhe_context
            del cache
        except Exception as exc:
            existing_ids = {str(row.get("case_id")) for row in results if row.get("case_id")}
            for case in cases:
                case_id = bench.case_id(case)
                if case_id in existing_ids:
                    continue
                results.append(
                    {
                        "case_id": case_id,
                        "op": case.op,
                        "status": "failed",
                        "reason": str(exc),
                        "traceback": traceback.format_exc(),
                        "cur_limbs": int(case.cur_limbs),
                        "dnum": int(case.dnum),
                        "slots": int(case.slots),
                        "ct_size": int(case.ct_size),
                        "extra": dict(case.extra),
                    }
                )
            failure_report = {
                "config": {
                    "device": str(args.device),
                    "context_max_levels_remaining": int(args.context_max_levels_remaining),
                    "target_limb_values": target_limb_values,
                    "bucket": {
                        "name": str(bucket["name"]),
                        "dnum": int(bucket["dnum"]),
                        "logbs": int(bucket["logbs"]),
                        "slots": int(bucket["slots"]),
                        "level_budget": [int(x) for x in bucket["level_budget"]],
                    },
                    "resume": not bool(args.no_resume),
                    "rerun_failed": bool(args.rerun_failed),
                },
                "case_manifest": [bench.case_manifest_row(case) for case in cases],
                "summary": {
                    "ok": sum(1 for row in results if row.get("status") == "ok"),
                    "unsupported": sum(1 for row in results if row.get("status") == "unsupported"),
                    "failed": sum(1 for row in results if row.get("status") == "failed"),
                },
                "results": results,
                "bucket_error": str(exc),
                "bucket_traceback": traceback.format_exc(),
            }
            _write_bucket_json(output_json, failure_report)
            run_rows.append(
                {
                    "bucket": {
                        "name": str(bucket["name"]),
                        "dnum": int(bucket["dnum"]),
                        "slots": int(bucket["slots"]),
                        "level_budget": [int(x) for x in bucket["level_budget"]],
                    },
                    "status": "failed",
                    "summary": dict(failure_report["summary"]),
                    "timed_output_json": str(output_json),
                    "reason": str(exc),
                }
            )
        finally:
            _release_cuda_memory()

    counts = {"ok": 0, "unsupported": 0, "failed": 0}
    for row in run_rows:
        counts["ok"] += int(row["summary"]["ok"])
        counts["unsupported"] += int(row["summary"]["unsupported"])
        counts["failed"] += int(row["summary"]["failed"])

    csv_update = _update_csv(args, out_dir)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    "device": str(args.device),
                    "context_max_levels_remaining": int(args.context_max_levels_remaining),
                    "target_limbs_min": int(args.target_limbs_min),
                    "target_limbs_max": {
                        str(raw): int(_effective_target_limbs_max(args, raw.split(","))) for raw in args.level_budgets
                    },
                    "dnum_values": [int(x) for x in args.dnum_values],
                    "logbs_values": [int(x) for x in args.logbs_values],
                    "level_budgets": [str(x) for x in args.level_budgets],
                    "warmup_heavy": int(args.warmup_heavy),
                    "timed_heavy": int(args.timed_heavy),
                    "bucket_count": len(bucket_defs),
                    "resume": not bool(args.no_resume),
                    "rerun_failed": bool(args.rerun_failed),
                },
                "counts": counts,
                "runs": run_rows,
                "csv_update": csv_update,
            },
            indent=2,
        )
    )

    print(
        json.dumps(
            {
                "summary_file": str(summary_path),
                "counts": counts,
                "csv_update": csv_update,
            },
            indent=2,
        )
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
