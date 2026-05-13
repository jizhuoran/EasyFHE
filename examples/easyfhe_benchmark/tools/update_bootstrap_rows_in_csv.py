from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


CSV_FIELDS = [
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


def _load_json(path: Path):
    return json.loads(path.read_text())


def _sort_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple("" if row.get(field, "") == "" else str(row.get(field, "")) for field in CSV_FIELDS[:-1])


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _allowed_timed_files(summary_json: Path | None) -> set[str] | None:
    if summary_json is None:
        return None
    data = _load_json(summary_json)
    allowed = set()
    for row in data.get("runs", []):
        if row.get("timed_status") == "ok":
            timed_output_json = str(row.get("timed_output_json", "")).strip()
            if timed_output_json:
                allowed.add(Path(timed_output_json).name)
    return allowed


def _bootstrap_rows_from_timed_dir(
    timed_dir: Path,
    *,
    accelerator: str,
    allowed_names: set[str] | None,
) -> List[Dict[str, str]]:
    rows: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for path in sorted(timed_dir.glob("timed__*.json")):
        if allowed_names is not None and path.name not in allowed_names:
            continue
        data = _load_json(path)
        if not isinstance(data, dict) or "results" not in data:
            continue
        cfg = data.get("config", {})
        n_value = str(1 << int(cfg.get("logN", 16)))
        for result in data.get("results", []):
            if result.get("op") != "bootstrap" or result.get("status") != "ok":
                continue
            timing = result.get("timing", {})
            median_us = timing.get("median_us")
            if median_us is None:
                continue
            extra = result.get("extra", {})
            level_budget = extra.get("level_budget", [0, 0])
            target_limbs = int(extra["target_limbs"])
            row = {
                "op": "bootstrap",
                "accelerator": str(accelerator),
                "N": n_value,
                "cur_limbs": str(int(result["cur_limbs"])),
                "target_limbs": str(target_limbs),
                "dnum": str(int(result["dnum"])),
                "slots": str(int(result["slots"])),
                "level_budget_0": str(int(level_budget[0]) if len(level_budget) >= 1 else 0),
                "level_budget_1": str(int(level_budget[1]) if len(level_budget) >= 2 else 0),
                "latency_us": f"{float(median_us):.6f}",
            }
            rows[_sort_key(row)] = row
    return [rows[key] for key in sorted(rows)]


def merge_rows(
    existing_rows: List[Dict[str, str]],
    bootstrap_rows: List[Dict[str, str]],
    *,
    replace_bootstrap: bool = False,
) -> List[Dict[str, str]]:
    merged: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in existing_rows:
        normalized = {field: row.get(field, "") for field in CSV_FIELDS}
        if replace_bootstrap and normalized.get("op") == "bootstrap":
            continue
        merged[_sort_key(normalized)] = normalized
    for row in bootstrap_rows:
        merged[_sort_key(row)] = row
    return [merged[key] for key in sorted(merged)]


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Update bootstrap rows in an existing profiling CSV without dropping existing rows")
    ap.add_argument("--timed-dir", required=True)
    ap.add_argument("--csv-in", required=True)
    ap.add_argument("--csv-out")
    ap.add_argument("--summary-json")
    ap.add_argument("--accelerator", default="sim")
    ap.add_argument("--replace-bootstrap", action="store_true")
    args = ap.parse_args()

    timed_dir = Path(args.timed_dir)
    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out) if args.csv_out else csv_in
    summary_json = Path(args.summary_json) if args.summary_json else None

    existing_rows = _read_csv_rows(csv_in)
    allowed_names = _allowed_timed_files(summary_json)
    bootstrap_rows = _bootstrap_rows_from_timed_dir(
        timed_dir,
        accelerator=str(args.accelerator),
        allowed_names=allowed_names,
    )
    merged_rows = merge_rows(
        existing_rows,
        bootstrap_rows,
        replace_bootstrap=bool(args.replace_bootstrap),
    )
    write_csv(csv_out, merged_rows)

    print(
        json.dumps(
            {
                "csv_in": str(csv_in),
                "csv_out": str(csv_out),
                "bootstrap_rows_added_or_updated": len(bootstrap_rows),
                "total_rows": len(merged_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
