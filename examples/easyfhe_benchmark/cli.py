from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    from .paths import BENCHMARK_ROOT, DATA_ROOT, PROJECT_ROOT, ensure_repo_on_path, repo_path
except ImportError:
    from paths import BENCHMARK_ROOT, DATA_ROOT, PROJECT_ROOT, ensure_repo_on_path, repo_path

ensure_repo_on_path()

CLI_PATH = BENCHMARK_ROOT / "cli.py"
BOOTSTRAP_BUCKETED_PATH = BENCHMARK_ROOT / "run_bootstrap_bucketed_profiling.py"
EXPORT_BASIC_PATH = BENCHMARK_ROOT / "tools" / "export_basic_ops_report_and_csv.py"
UPDATE_BOOTSTRAP_CSV_PATH = BENCHMARK_ROOT / "tools" / "update_bootstrap_rows_in_csv.py"
PRESET_CHOICES = [
    "bootstrap-profiling",
    "full-profiling",
    "full-pipeline",
]

BASIC_PROFILED_OPS: List[str] = [
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

FULL_PROFILING_OPS: List[str] = list(BASIC_PROFILED_OPS)

DEFAULT_BOOTSTRAP_SECRET_KEY_DIST = "SPARSE_TERNARY"


def _default_python_bin() -> str:
    if sys.executable:
        return sys.executable
    return shutil.which("python3") or "python3"


def _default_save_dir() -> str:
    return str(DATA_ROOT / "context_store")


def _default_device() -> str:
    probe_cmd = [
        _default_python_bin(),
        "-c",
        "import easyfhe as torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
    ]
    try:
        proc = subprocess.run(
            probe_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "cpu"
    detected = (proc.stdout or "").strip().lower()
    return detected if detected in {"cpu", "cuda"} else "cpu"


def _run_subprocess(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), cwd=PROJECT_ROOT, capture_output=True, text=True)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _repo_path(path_like: str | Path) -> Path:
    return repo_path(path_like)


def _cleanup_context_cache(mode: str, root: Path | None, *, keep: bool = False, dry_run: bool = False) -> None:
    if keep or dry_run or str(mode) != "transient" or root is None or not root.exists():
        return
    shutil.rmtree(root, ignore_errors=True)


def _echo_subprocess(label: str, cmd: Sequence[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"[{label}] {rendered}")
    if dry_run:
        return subprocess.CompletedProcess(list(cmd), 0, "", "")
    proc = _run_subprocess(cmd)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        print(stdout)
    if proc.returncode != 0 and stderr:
        print(stderr[-4000:], file=sys.stderr)
    return proc


def run_timed_profile_once(
    *,
    python_bin: str,
    output_json: Path,
    save_dir: str,
    ops: Sequence[str],
    max_levels_remaining: int,
    limb_min: int,
    limb_max: int,
    basic_slots: int,
    encode_slot_values: Sequence[int],
    dnum_values: Sequence[int],
    default_dnum: int,
    rotate_steps: Sequence[int],
    context_cache_mode: str = "persistent",
    context_cache_root: str | None = None,
    device: str = "cuda",
    warmup_cheap: int = 30,
    timed_cheap: int = 100,
    warmup_medium: int = 20,
    timed_medium: int = 50,
    timing_modes: Sequence[str] | None = None,
    batched_group_size: int = 30,
    rotation_random_mode: str = "reuse_by_shape",
    resume: bool = True,
    rerun_failed: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        python_bin,
        str(BENCHMARK_ROOT / "timed_harness.py"),
        "--output-json",
        str(output_json),
        "--save-dir",
        str(save_dir),
        "--context-cache-mode",
        str(context_cache_mode),
        "--ops",
        *[str(op) for op in ops],
        "--device",
        str(device),
        "--max-levels-remaining",
        str(max_levels_remaining),
        "--limb-min",
        str(limb_min),
        "--limb-max",
        str(limb_max),
        "--basic-slots",
        str(basic_slots),
        "--encode-slot-values",
        *[str(value) for value in encode_slot_values],
        "--dnum-values",
        *[str(v) for v in dnum_values],
        "--default-dnum",
        str(default_dnum),
        "--rotate-steps",
        *[str(v) for v in rotate_steps],
        "--warmup-cheap",
        str(warmup_cheap),
        "--timed-cheap",
        str(timed_cheap),
        "--warmup-medium",
        str(warmup_medium),
        "--timed-medium",
        str(timed_medium),
        "--rotation-random-mode",
        str(rotation_random_mode),
    ]
    if context_cache_root:
        cmd.extend(["--context-cache-root", str(context_cache_root)])
    if timing_modes:
        cmd.extend([
            "--timing-modes",
            *[str(mode) for mode in timing_modes],
        ])
    if not resume:
        cmd.append("--no-resume")
    if rerun_failed:
        cmd.append("--rerun-failed")
    cmd.extend([
        "--batched-group-size",
        str(int(batched_group_size)),
    ])
    return _run_subprocess(cmd)


def run_bootstrap_profiling_main(argv: Sequence[str] | None = None) -> int:
    from easyfhe_benchmark.run_bootstrap_bucketed_profiling import main as bucketed_main

    return bucketed_main(argv)


def _aggregate_timed_by_op(run_rows: List[Dict]) -> Dict[str, Dict]:
    by_op = defaultdict(
        lambda: {
            "runs": 0,
            "ok": 0,
            "unsupported": 0,
            "failed": 0,
            "median_values": [],
            "batched_median_values": [],
        }
    )
    for row in run_rows:
        output_json = row.get("output_json")
        if not output_json:
            continue
        data = _load_json(Path(output_json))
        per_run_seen = set()
        for item in data.get("results", []):
            op = str(item.get("op"))
            status = str(item.get("status"))
            if status not in {"ok", "unsupported", "failed"}:
                continue
            by_op[op][status] += 1
            per_run_seen.add(op)
            timing = item.get("timing", {})
            if status == "ok" and "median_us" in timing:
                by_op[op]["median_values"].append(float(timing["median_us"]))
            batched_timing = timing.get("batched", {})
            if status == "ok" and "median_us" in batched_timing:
                by_op[op]["batched_median_values"].append(float(batched_timing["median_us"]))
        for op in per_run_seen:
            by_op[op]["runs"] += 1
    return {
        op: {
            "runs": int(stats["runs"]),
            "ok": int(stats["ok"]),
            "unsupported": int(stats["unsupported"]),
            "failed": int(stats["failed"]),
            "median_of_medians_us": float(np.median(stats["median_values"])) if stats["median_values"] else None,
            "isolated_median_of_medians_us": float(np.median(stats["median_values"])) if stats["median_values"] else None,
            "batched_median_of_medians_us": float(np.median(stats["batched_median_values"]))
            if stats["batched_median_values"]
            else None,
        }
        for op, stats in by_op.items()
    }


def run_full_profiling_main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run timed profiling for foundational EasyFHE ops")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--python-bin", default=_default_python_bin())
    ap.add_argument("--ops", nargs="+", default=FULL_PROFILING_OPS)
    ap.add_argument("--limb-min", type=int, default=1)
    ap.add_argument("--limb-max", type=int, default=30)
    ap.add_argument("--basic-slots", type=int, default=4096)
    ap.add_argument("--encode-slot-values", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    ap.add_argument("--max-levels-remaining", type=int, default=39)
    ap.add_argument("--dnum-values", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--default-dnum", type=int, default=3)
    ap.add_argument("--rotate-steps", nargs="+", type=int, default=[-1])
    ap.add_argument("--warmup-cheap", type=int, default=30)
    ap.add_argument("--timed-cheap", type=int, default=100)
    ap.add_argument("--warmup-medium", type=int, default=20)
    ap.add_argument("--timed-medium", type=int, default=50)
    ap.add_argument("--timing-modes", nargs="+", choices=["isolated", "batched"], default=["isolated", "batched"])
    ap.add_argument("--batched-group-size", type=int, default=30)
    ap.add_argument("--csv-timing-mode", choices=["isolated", "batched"], default="isolated")
    ap.add_argument("--save-dir", default=_default_save_dir())
    ap.add_argument("--output-dir", default=str(DATA_ROOT / "full_profiling"))
    ap.add_argument("--summary-json")
    ap.add_argument("--report-md")
    ap.add_argument("--csv-out")
    ap.add_argument("--accelerator", default="sim")
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--rotation-random-mode", choices=["fresh", "reuse_by_shape"], default="reuse_by_shape")
    ap.add_argument("--context-cache-mode", choices=["persistent", "transient", "memory"], default="transient")
    ap.add_argument("--context-cache-root")
    ap.add_argument("--keep-context-cache", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--rerun-failed", action="store_true")
    args = ap.parse_args(argv)

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    dnum_values = sorted({int(v) for v in args.dnum_values})
    if not dnum_values:
        raise ValueError("--dnum-values must not be empty")
    encode_slot_values = sorted({int(v) for v in args.encode_slot_values})
    if not encode_slot_values:
        raise ValueError("--encode-slot-values must not be empty")

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_cache_root = (
        _repo_path(args.context_cache_root) if args.context_cache_root else output_dir / ".context_cache"
    ) if str(args.context_cache_mode) == "transient" else None

    requested_ops = [str(op) for op in args.ops]
    timed_ops = [op for op in requested_ops if op in BASIC_PROFILED_OPS]

    timed_runs: List[Dict] = []
    try:
        for run_idx in range(1, args.repeats + 1):
            timed_output_json = output_dir / f"timed_basic_ops__run{run_idx}.json"
            timed_proc = run_timed_profile_once(
                python_bin=str(args.python_bin),
                output_json=timed_output_json,
                save_dir=str(args.save_dir),
                context_cache_mode=str(args.context_cache_mode),
                context_cache_root=str(context_cache_root) if context_cache_root else None,
                ops=timed_ops,
                max_levels_remaining=int(args.max_levels_remaining),
                limb_min=int(args.limb_min),
                limb_max=int(args.limb_max),
                basic_slots=int(args.basic_slots),
                encode_slot_values=encode_slot_values,
                dnum_values=dnum_values,
                default_dnum=int(args.default_dnum),
                rotate_steps=[int(v) for v in args.rotate_steps],
                device=str(args.device),
                warmup_cheap=int(args.warmup_cheap),
                timed_cheap=int(args.timed_cheap),
                warmup_medium=int(args.warmup_medium),
                timed_medium=int(args.timed_medium),
                timing_modes=[str(mode) for mode in args.timing_modes],
                batched_group_size=int(args.batched_group_size),
                rotation_random_mode=str(args.rotation_random_mode),
                resume=not bool(args.no_resume),
                rerun_failed=bool(args.rerun_failed),
            )
            if not timed_output_json.exists():
                timed_runs.append(
                    {
                        "op": "all",
                        "run": run_idx,
                        "status": "failed",
                        "summary": {"ok": 0, "unsupported": 0, "failed": 1},
                        "reason": "missing timed output json",
                        "exit_code": timed_proc.returncode,
                        "stderr_tail": (timed_proc.stderr or "")[-1200:],
                    }
                )
                continue
            timed_data = _load_json(timed_output_json)
            timed_summary = timed_data.get("summary", {})
            timed_runs.append(
                {
                    "op": "all",
                    "run": run_idx,
                    "status": "ok" if int(timed_summary.get("failed", 0)) == 0 else "failed",
                    "summary": {
                        "ok": int(timed_summary.get("ok", 0)),
                        "unsupported": int(timed_summary.get("unsupported", 0)),
                        "failed": int(timed_summary.get("failed", 0)),
                    },
                    "reason": "",
                    "exit_code": timed_proc.returncode,
                    "stderr_tail": (timed_proc.stderr or "")[-1200:],
                    "output_json": str(timed_output_json),
                }
            )
    finally:
        _cleanup_context_cache(
            str(args.context_cache_mode),
            context_cache_root,
            keep=bool(args.keep_context_cache),
        )

    timed_aggregate = _aggregate_timed_by_op(timed_runs)
    summary = {
        "config": {
            "repeats": int(args.repeats),
            "requested_ops": requested_ops,
            "timed_ops": timed_ops,
            "limb_min": int(args.limb_min),
            "limb_max": int(args.limb_max),
            "basic_slots": int(args.basic_slots),
            "encode_slot_values": encode_slot_values,
            "max_levels_remaining": int(args.max_levels_remaining),
            "dnum_values": dnum_values,
            "default_dnum": int(args.default_dnum),
            "rotate_steps": [int(v) for v in args.rotate_steps],
            "warmup_cheap": int(args.warmup_cheap),
            "timed_cheap": int(args.timed_cheap),
            "warmup_medium": int(args.warmup_medium),
            "timed_medium": int(args.timed_medium),
            "timing_modes": [str(mode) for mode in args.timing_modes],
            "batched_group_size": int(args.batched_group_size),
            "csv_timing_mode": str(args.csv_timing_mode),
            "context_cache_mode": str(args.context_cache_mode),
            "resume": not bool(args.no_resume),
            "rerun_failed": bool(args.rerun_failed),
            "rotation_random_mode": str(args.rotation_random_mode),
        },
        "timed_runs": timed_runs,
        "timed_aggregate": timed_aggregate,
    }

    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    report_path = Path(args.report_md) if args.report_md else output_dir / "REPORT.md"
    csv_path = Path(args.csv_out) if args.csv_out else output_dir / "easyfhe_latency_full_profiled.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        str(args.python_bin),
        str(EXPORT_BASIC_PATH),
        "--summary-json",
        str(summary_path),
        "--report-md",
        str(report_path),
        "--csv-out",
        str(csv_path),
        "--accelerator",
        str(args.accelerator),
        "--timing-mode",
        str(args.csv_timing_mode),
    ]
    export_proc = _run_subprocess(export_cmd)

    output = {
        "summary_file": str(summary_path),
        "report_file": str(report_path),
        "csv_file": str(csv_path),
        "export_exit_code": int(export_proc.returncode),
        "aggregate_ops": sorted(timed_aggregate.keys()),
    }
    print(json.dumps(output, indent=2))

    any_timed_fail = any(int(row["summary"]["failed"]) > 0 for row in timed_runs)
    return 0 if (not any_timed_fail and export_proc.returncode == 0) else 1


def run_full_pipeline_main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-shot foundational + bootstrap profiling pipeline")
    ap.add_argument("--python-bin", default=_default_python_bin())
    ap.add_argument("--save-dir", default=_default_save_dir())
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--accelerator", default="sim")
    ap.add_argument("--rotation-random-mode", choices=["fresh", "reuse_by_shape"], default="reuse_by_shape")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-batched-export", action="store_true")
    ap.add_argument("--context-cache-mode", choices=["persistent", "transient", "memory"], default="transient")
    ap.add_argument("--context-cache-root")
    ap.add_argument("--keep-context-cache", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--rerun-failed", action="store_true")

    ap.add_argument("--output-dir", default=str(DATA_ROOT / "full_profiling"))
    ap.add_argument("--summary-json")
    ap.add_argument("--report-md")
    ap.add_argument("--csv-out")
    ap.add_argument("--pipeline-summary-json")

    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ops", nargs="+", default=FULL_PROFILING_OPS)
    ap.add_argument("--limb-min", type=int, default=1)
    ap.add_argument("--limb-max", type=int, default=40)
    ap.add_argument("--basic-slots", type=int, default=4096)
    ap.add_argument("--encode-slot-values", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    ap.add_argument("--max-levels-remaining", type=int, default=39)
    ap.add_argument("--dnum-values", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--default-dnum", type=int, default=3)
    ap.add_argument("--rotate-steps", nargs="+", type=int, default=[-1])
    ap.add_argument("--warmup-cheap", type=int, default=30)
    ap.add_argument("--timed-cheap", type=int, default=100)
    ap.add_argument("--warmup-medium", type=int, default=20)
    ap.add_argument("--timed-medium", type=int, default=50)
    ap.add_argument("--timing-modes", nargs="+", choices=["isolated", "batched"], default=["isolated", "batched"])
    ap.add_argument("--batched-group-size", type=int, default=30)
    ap.add_argument("--csv-timing-mode", choices=["isolated", "batched"], default="isolated")

    ap.add_argument("--bootstrap-output-dir", default=str(DATA_ROOT / "bootstrap_bucketed_fullscan"))
    ap.add_argument("--bootstrap-context-max-levels-remaining", type=int, default=25)
    ap.add_argument("--bootstrap-target-limbs-min", type=int, default=1)
    ap.add_argument("--bootstrap-target-limbs-max", type=int)
    ap.add_argument("--bootstrap-dnum-values", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--bootstrap-logbs-values", nargs="+", type=int, default=[12, 13, 14])
    ap.add_argument("--bootstrap-level-budgets", nargs="+", default=["3,3", "4,4"])
    ap.add_argument("--bootstrap-warmup-heavy", type=int, default=1)
    ap.add_argument("--bootstrap-timed-heavy", type=int, default=3)
    ap.add_argument("--bootstrap-logN", type=int, default=16)
    ap.add_argument("--bootstrap-dcrt-bits", type=int, default=52)
    ap.add_argument("--bootstrap-first-mod", type=int, default=55)
    ap.add_argument("--bootstrap-secret-key-dist", default=DEFAULT_BOOTSTRAP_SECRET_KEY_DIST)
    ap.add_argument("--bootstrap-rescale-tech", default="FIXEDMANUAL")

    ap.add_argument("--batched-report-md")
    ap.add_argument("--batched-basic-csv-out")
    ap.add_argument("--batched-csv-out")
    args = ap.parse_args(argv)

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_cache_root = (
        _repo_path(args.context_cache_root) if args.context_cache_root else output_dir / ".context_cache"
    ) if str(args.context_cache_mode) == "transient" else None

    summary_json = _repo_path(args.summary_json) if args.summary_json else output_dir / "summary.json"
    report_md = _repo_path(args.report_md) if args.report_md else output_dir / "REPORT.md"
    csv_out = _repo_path(args.csv_out) if args.csv_out else output_dir / "easyfhe_latency_full_profiled.csv"
    pipeline_summary_json = (
        _repo_path(args.pipeline_summary_json) if args.pipeline_summary_json else output_dir / "pipeline_summary.json"
    )

    bootstrap_output_dir = _repo_path(args.bootstrap_output_dir)

    batched_report_md = (
        _repo_path(args.batched_report_md) if args.batched_report_md else output_dir / "REPORT.batched_basic.md"
    )
    batched_basic_csv_out = (
        _repo_path(args.batched_basic_csv_out)
        if args.batched_basic_csv_out
        else output_dir / "easyfhe_latency_full_profiled.batched_basic_only.csv"
    )
    batched_csv_out = (
        _repo_path(args.batched_csv_out) if args.batched_csv_out else output_dir / "easyfhe_latency_full_profiled.batched.csv"
    )

    pipeline_summary = {
        "config": {
            "python_bin": str(args.python_bin),
            "device": str(args.device),
            "accelerator": str(args.accelerator),
            "csv_timing_mode": str(args.csv_timing_mode),
            "skip_batched_export": bool(args.skip_batched_export),
            "context_cache_mode": str(args.context_cache_mode),
            "resume": not bool(args.no_resume),
            "rerun_failed": bool(args.rerun_failed),
            "rotation_random_mode": str(args.rotation_random_mode),
        },
        "outputs": {
            "summary_json": str(summary_json),
            "report_md": str(report_md),
            "csv_out": str(csv_out),
            "bootstrap_output_dir": str(bootstrap_output_dir),
            "batched_report_md": str(batched_report_md),
            "batched_basic_csv_out": str(batched_basic_csv_out),
            "batched_csv_out": str(batched_csv_out),
        },
        "steps": [],
    }

    foundational_cmd = [
        str(args.python_bin),
        str(CLI_PATH),
        "--preset",
        "full-profiling",
        "--python-bin",
        str(args.python_bin),
        "--repeats",
        str(int(args.repeats)),
        "--ops",
        *[str(op) for op in args.ops],
        "--limb-min",
        str(int(args.limb_min)),
        "--limb-max",
        str(int(args.limb_max)),
        "--basic-slots",
        str(int(args.basic_slots)),
        "--encode-slot-values",
        *[str(value) for value in args.encode_slot_values],
        "--max-levels-remaining",
        str(int(args.max_levels_remaining)),
        "--dnum-values",
        *[str(value) for value in args.dnum_values],
        "--default-dnum",
        str(int(args.default_dnum)),
        "--rotate-steps",
        *[str(value) for value in args.rotate_steps],
        "--warmup-cheap",
        str(int(args.warmup_cheap)),
        "--timed-cheap",
        str(int(args.timed_cheap)),
        "--warmup-medium",
        str(int(args.warmup_medium)),
        "--timed-medium",
        str(int(args.timed_medium)),
        "--timing-modes",
        *[str(mode) for mode in args.timing_modes],
        "--batched-group-size",
        str(int(args.batched_group_size)),
        "--csv-timing-mode",
        str(args.csv_timing_mode),
        "--save-dir",
        str(args.save_dir),
        "--context-cache-mode",
        str(args.context_cache_mode),
        "--output-dir",
        str(output_dir),
        "--summary-json",
        str(summary_json),
        "--report-md",
        str(report_md),
        "--csv-out",
        str(csv_out),
        "--accelerator",
        str(args.accelerator),
        "--device",
        str(args.device),
        "--rotation-random-mode",
        str(args.rotation_random_mode),
    ]
    if args.no_resume:
        foundational_cmd.append("--no-resume")
    if args.rerun_failed:
        foundational_cmd.append("--rerun-failed")
    if context_cache_root:
        foundational_cmd.extend(["--context-cache-root", str(context_cache_root)])
    if args.keep_context_cache:
        foundational_cmd.append("--keep-context-cache")
    bootstrap_cmd = [
        str(args.python_bin),
        str(BOOTSTRAP_BUCKETED_PATH),
        "--save-dir",
        str(args.save_dir),
        "--context-cache-mode",
        str(args.context_cache_mode),
        "--output-dir",
        str(bootstrap_output_dir),
        "--csv-inout",
        str(csv_out),
        "--accelerator",
        str(args.accelerator),
        "--device",
        str(args.device),
        "--context-max-levels-remaining",
        str(int(args.bootstrap_context_max_levels_remaining)),
        "--target-limbs-min",
        str(int(args.bootstrap_target_limbs_min)),
        "--dnum-values",
        *[str(value) for value in args.bootstrap_dnum_values],
        "--logbs-values",
        *[str(value) for value in args.bootstrap_logbs_values],
        "--level-budgets",
        *[str(value) for value in args.bootstrap_level_budgets],
        "--warmup-heavy",
        str(int(args.bootstrap_warmup_heavy)),
        "--timed-heavy",
        str(int(args.bootstrap_timed_heavy)),
        "--logN",
        str(int(args.bootstrap_logN)),
        "--dcrt-bits",
        str(int(args.bootstrap_dcrt_bits)),
        "--first-mod",
        str(int(args.bootstrap_first_mod)),
        "--secret-key-dist",
        str(args.bootstrap_secret_key_dist),
        "--rescale-tech",
        str(args.bootstrap_rescale_tech),
        "--rotation-random-mode",
        str(args.rotation_random_mode),
    ]
    if context_cache_root:
        bootstrap_cmd.extend(["--context-cache-root", str(context_cache_root)])
    if args.bootstrap_target_limbs_max is not None:
        bootstrap_cmd.extend(["--target-limbs-max", str(int(args.bootstrap_target_limbs_max))])
    if args.no_resume:
        bootstrap_cmd.append("--no-resume")
    if args.rerun_failed:
        bootstrap_cmd.append("--rerun-failed")
    try:
        foundational_proc = _echo_subprocess("full-pipeline:foundational", foundational_cmd, dry_run=bool(args.dry_run))
        pipeline_summary["steps"].append(
            {"name": "foundational", "returncode": int(foundational_proc.returncode), "command": foundational_cmd}
        )
        if foundational_proc.returncode != 0:
            if not args.dry_run:
                pipeline_summary_json.write_text(json.dumps(pipeline_summary, indent=2))
            return 1

        bootstrap_proc = _echo_subprocess("full-pipeline:bootstrap", bootstrap_cmd, dry_run=bool(args.dry_run))
        pipeline_summary["steps"].append(
            {"name": "bootstrap", "returncode": int(bootstrap_proc.returncode), "command": bootstrap_cmd}
        )
        if bootstrap_proc.returncode != 0:
            if not args.dry_run:
                pipeline_summary_json.write_text(json.dumps(pipeline_summary, indent=2))
            return 1

        if not args.skip_batched_export:
            batched_basic_cmd = [
                str(args.python_bin),
                str(EXPORT_BASIC_PATH),
                "--summary-json",
                str(summary_json),
                "--report-md",
                str(batched_report_md),
                "--csv-out",
                str(batched_basic_csv_out),
                "--accelerator",
                str(args.accelerator),
                "--timing-mode",
                "batched",
            ]
            batched_basic_proc = _echo_subprocess(
                "full-pipeline:batched-basic-export",
                batched_basic_cmd,
                dry_run=bool(args.dry_run),
            )
            pipeline_summary["steps"].append(
                {
                    "name": "batched-basic-export",
                    "returncode": int(batched_basic_proc.returncode),
                    "command": batched_basic_cmd,
                }
            )
            if batched_basic_proc.returncode != 0:
                if not args.dry_run:
                    pipeline_summary_json.write_text(json.dumps(pipeline_summary, indent=2))
                return 1

            batched_merge_cmd = [
                str(args.python_bin),
                str(UPDATE_BOOTSTRAP_CSV_PATH),
                "--timed-dir",
                str(bootstrap_output_dir),
                "--csv-in",
                str(batched_basic_csv_out),
                "--csv-out",
                str(batched_csv_out),
                "--accelerator",
                str(args.accelerator),
            ]
            batched_merge_proc = _echo_subprocess(
                "full-pipeline:batched-merge",
                batched_merge_cmd,
                dry_run=bool(args.dry_run),
            )
            pipeline_summary["steps"].append(
                {"name": "batched-merge", "returncode": int(batched_merge_proc.returncode), "command": batched_merge_cmd}
            )
            if batched_merge_proc.returncode != 0:
                if not args.dry_run:
                    pipeline_summary_json.write_text(json.dumps(pipeline_summary, indent=2))
                return 1

        if not args.dry_run:
            pipeline_summary_json.write_text(json.dumps(pipeline_summary, indent=2))
    finally:
        _cleanup_context_cache(
            str(args.context_cache_mode),
            context_cache_root,
            keep=bool(args.keep_context_cache),
            dry_run=bool(args.dry_run),
        )
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "report_md": str(report_md),
                "csv_out": str(csv_out),
                "bootstrap_output_dir": str(bootstrap_output_dir),
                "batched_csv_out": None if args.skip_batched_export else str(batched_csv_out),
                "pipeline_summary_json": str(pipeline_summary_json),
            },
            indent=2,
        )
    )
    return 0


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]

    dispatch = {
        "bootstrap-profiling": run_bootstrap_profiling_main,
        "full-profiling": run_full_profiling_main,
        "full-pipeline": run_full_pipeline_main,
    }

    top_ap = argparse.ArgumentParser(description="EasyFHE benchmark entrypoint with presets", add_help=False)
    top_ap.add_argument("--preset", choices=PRESET_CHOICES)
    top_args, rest = top_ap.parse_known_args(argv)

    if top_args.preset is None:
        ap = argparse.ArgumentParser(description="EasyFHE benchmark entrypoint with presets")
        ap.add_argument("--preset", choices=PRESET_CHOICES, required=True)
        ap.parse_args(argv)
        return 0

    if any(flag in rest for flag in ("-h", "--help")):
        return dispatch[top_args.preset](["--help"])

    args = top_args
    if args.preset == "bootstrap-profiling":
        return run_bootstrap_profiling_main(rest)
    if args.preset == "full-profiling":
        return run_full_profiling_main(rest)
    if args.preset == "full-pipeline":
        return run_full_pipeline_main(rest)
    raise ValueError(f"Unsupported preset: {args.preset}")


__all__ = [
    "BASIC_PROFILED_OPS",
    "FULL_PROFILING_OPS",
    "benchmark_main",
    "run_bootstrap_profiling_main",
    "run_full_profiling_main",
    "run_full_pipeline_main",
]


if __name__ == "__main__":
    raise SystemExit(benchmark_main())
