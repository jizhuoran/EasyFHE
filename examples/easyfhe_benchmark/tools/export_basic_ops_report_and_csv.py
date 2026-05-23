from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Seed latencies for simulator-compatible CSV export.
# These are placeholders until timed profiling data is available.
DEFAULT_LATENCY_US = {
    "encrypt": 0.0,
    "decrypt": 0.0,
    "expand_slots": 0.0,
    "clone": 0.0,
    "encode": 4.0,
    "add": 3.0,
    "add_pt": 2.0,
    "mul_pt": 30.0,
    "mul": 120.0,
    "rescale": 60.0,
    "force_rescale": 8.0,
    "drop_last_elements": 1.0,
    "modup_to_ext": 5.0,
    "moddown_from_ext": 5.5,
    "square": 120.0,
    "rotate": 80.0,
    "fast_rotate": 45.0,
    "bootstrap": 900.0,
}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _target_limbs(op: str, cur_limbs: int, details: Dict, extra: Dict) -> int:
    if "result_cur_limbs" in details:
        return int(details["result_cur_limbs"])
    if op in {"rescale", "force_rescale", "drop_last_elements"}:
        return int(max(cur_limbs - 1, 1))
    return int(cur_limbs)


def _sort_key(row_key: Tuple) -> Tuple[str, ...]:
    return tuple("" if value == "" else str(value) for value in row_key)


def _seed_zero_cost_rows(summary: Dict, accelerator: str) -> Dict[Tuple, float]:
    return {
        (op, accelerator, "", "", "", "", "", "", ""): float(DEFAULT_LATENCY_US[op])
        for op in ("encrypt", "decrypt", "expand_slots", "clone")
    }


def _timing_value_for_mode(timing: Dict, timing_mode: str) -> float | None:
    if timing_mode == "isolated":
        median_us = timing.get("median_us")
        if median_us is not None:
            return float(median_us)
        isolated = timing.get("isolated", {})
        if isolated.get("median_us") is not None:
            return float(isolated["median_us"])
        return None

    batched = timing.get("batched", {})
    if batched.get("median_us") is not None:
        return float(batched["median_us"])
    return None


def collect_supported_rows(summary: Dict, accelerator: str, timing_mode: str) -> List[Tuple[Tuple, float]]:
    timing_run_items = summary.get("timed_runs", summary.get("runs", []))
    run1_files = [
        Path(item["output_json"])
        for item in timing_run_items
        if int(item.get("run", 0)) == 1 and item.get("output_json")
    ]
    rows: Dict[Tuple, float] = _seed_zero_cost_rows(summary, accelerator)
    measured_rows: Dict[Tuple, List[float]] = {}
    for file_path in run1_files:
        data = load_json(file_path)
        cfg = data.get("config", {})
        n_value = 1 << int(cfg.get("logN", 16))
        for row in data.get("results", []):
            if row.get("status") != "ok":
                continue
            op = str(row["op"])
            cur_limbs = int(row["cur_limbs"])
            dnum = int(row.get("dnum", cfg.get("default_dnum", 3)))
            slots = int(row["slots"])
            details = row.get("details", {})
            extra = row.get("extra", {})
            level_budget = extra.get("level_budget", [0, 0])
            lb0 = int(level_budget[0]) if len(level_budget) >= 1 else 0
            lb1 = int(level_budget[1]) if len(level_budget) >= 2 else 0
            target_limbs = _target_limbs(op, cur_limbs, details, extra)
            key = (
                op,
                accelerator,
                n_value,
                cur_limbs,
                target_limbs,
                dnum,
                slots,
                lb0,
                lb1,
            )
            timing = row.get("timing", {})
            median_us = _timing_value_for_mode(timing, timing_mode)
            if median_us is not None:
                measured_rows.setdefault(key, []).append(float(median_us))
            elif key not in rows:
                rows[key] = float(DEFAULT_LATENCY_US.get(op, 0.0))
    for key, values in measured_rows.items():
        rows[key] = float(np.median(np.asarray(values, dtype=np.float64)))
    return sorted(rows.items(), key=lambda kv: _sort_key(kv[0]))


def write_csv(path: Path, rows: List[Tuple[Tuple, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "op",
                "accelerator",
                "N",
                "cur_limbs",
                "target_limbs",
                "dnum",
                "slots",
                "level_budget_0",
                "level_budget_1",
                "latency_us",
            ]
        )
        for key, latency_us in rows:
            writer.writerow([*key, f"{latency_us:.6f}"])


def write_report(path: Path, summary: Dict, rows: List[Tuple[Tuple, float]], csv_path: Path, timing_mode: str) -> None:
    aggregate = summary.get("timed_aggregate", summary.get("aggregate", {}))
    lines: List[str] = []
    lines.append("# Basic Ops Batch Report")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    cfg = summary["config"]
    lines.append(f"- repeats: {cfg['repeats']}")
    lines.append(f"- limb range: {cfg['limb_min']}..{cfg['limb_max']}")
    lines.append(f"- basic_slots: {cfg['basic_slots']}")
    encode_slot_values = cfg.get("encode_slot_values")
    if encode_slot_values is not None:
        lines.append(f"- encode_slot_values: {', '.join(str(value) for value in encode_slot_values)}")
    elif "encode_slots" in cfg:
        lines.append(f"- encode_slots: {cfg['encode_slots']}")
    lines.append(f"- timed ops: {', '.join(cfg.get('timed_ops', cfg.get('ops', [])))}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| op | runs | ok | unsupported | failed | isolated_median_of_medians_us | batched_median_of_medians_us |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for op in sorted(aggregate):
        row = aggregate[op]
        isolated_value = row.get("isolated_median_of_medians_us", row.get("median_of_medians_us"))
        isolated_text = "" if isolated_value is None else f"{float(isolated_value):.3f}"
        batched_value = row.get("batched_median_of_medians_us")
        batched_text = "" if batched_value is None else f"{float(batched_value):.3f}"
        lines.append(
            f"| {op} | {row['runs']} | {row['ok']} | {row['unsupported']} | {row['failed']} | {isolated_text} | {batched_text} |"
        )

    total_failed = sum(int(row["failed"]) for row in aggregate.values())
    total_unsupported = sum(int(row["unsupported"]) for row in aggregate.values())
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- CSV timing mode: {timing_mode}")
    lines.append(f"- total failed tuples across repeats: {total_failed}")
    lines.append(f"- total unsupported tuples across repeats: {total_unsupported}")
    lines.append(f"- simulator CSV exported to: {csv_path}")
    lines.append("- CSV latencies come from timed profiling medians for the selected timing mode when available; fixed-zero ops are seeded directly.")
    lines.append(f"- supported tuple count exported: {len(rows)}")
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export basic-op batch report and simulator CSV")
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--report-md", required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--accelerator", default="sim")
    ap.add_argument("--timing-mode", choices=["isolated", "batched"], default="isolated")
    args = ap.parse_args()

    summary_path = Path(args.summary_json)
    report_path = Path(args.report_md)
    csv_path = Path(args.csv_out)

    summary = load_json(summary_path)
    rows = collect_supported_rows(summary, args.accelerator, args.timing_mode)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(csv_path, rows)
    write_report(report_path, summary, rows, csv_path, args.timing_mode)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "csv": str(csv_path),
                "rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
