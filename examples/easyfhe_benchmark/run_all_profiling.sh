#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$BENCHMARK_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[run_all_profiling] ERROR: cannot find python interpreter; set PYTHON_BIN explicitly"
  exit 1
fi

if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$("$PYTHON_BIN" -c "import easyfhe as torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || printf 'cpu')"
fi

FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-$BENCHMARK_DIR/data/full_profiling}"
FULL_SUMMARY_JSON="${FULL_SUMMARY_JSON:-$FULL_OUTPUT_DIR/summary.json}"
FULL_REPORT_MD="${FULL_REPORT_MD:-$FULL_OUTPUT_DIR/REPORT.md}"
FULL_CSV_OUT="${FULL_CSV_OUT:-$FULL_OUTPUT_DIR/easyfhe_latency_full_profiled.csv}"
PIPELINE_SUMMARY_JSON="${PIPELINE_SUMMARY_JSON:-$FULL_OUTPUT_DIR/pipeline_summary.json}"

BOOTSTRAP_OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR:-$BENCHMARK_DIR/data/bootstrap_bucketed_fullscan}"
SAVE_DIR="${SAVE_DIR:-$BENCHMARK_DIR/data/context_store}"
CONTEXT_CACHE_MODE="${CONTEXT_CACHE_MODE:-transient}"

BATCHED_REPORT_MD="${BATCHED_REPORT_MD:-$FULL_OUTPUT_DIR/REPORT.batched_basic.md}"
BATCHED_BASIC_CSV_OUT="${BATCHED_BASIC_CSV_OUT:-$FULL_OUTPUT_DIR/easyfhe_latency_full_profiled.batched_basic_only.csv}"
BATCHED_CSV_OUT="${BATCHED_CSV_OUT:-$FULL_OUTPUT_DIR/easyfhe_latency_full_profiled.batched.csv}"

mkdir -p "$FULL_OUTPUT_DIR" "$BOOTSTRAP_OUTPUT_DIR" "$SAVE_DIR"

cmd=(
  "$PYTHON_BIN"
  "$BENCHMARK_DIR/cli.py"
  --preset full-pipeline
  --python-bin "$PYTHON_BIN"
  --save-dir "$SAVE_DIR"
  --context-cache-mode "$CONTEXT_CACHE_MODE"
  --device "$DEVICE"
  --accelerator "${ACCELERATOR:-sim}"
  --output-dir "$FULL_OUTPUT_DIR"
  --summary-json "$FULL_SUMMARY_JSON"
  --report-md "$FULL_REPORT_MD"
  --csv-out "$FULL_CSV_OUT"
  --pipeline-summary-json "$PIPELINE_SUMMARY_JSON"
  --repeats "${FULL_REPEATS:-1}"
  --ops ${FULL_OPS:-encode add add_pt mul_pt mul rescale force_rescale drop_last_elements modup_to_ext moddown_from_ext square rotate eval_fast_rotate}
  --limb-min "${FULL_LIMB_MIN:-1}"
  --limb-max "${FULL_LIMB_MAX:-40}"
  --basic-slots "${FULL_BASIC_SLOTS:-4096}"
  --encode-slot-values ${FULL_ENCODE_SLOT_VALUES:-4096 8192 16384 32768}
  --max-levels-remaining "${FULL_MAX_LEVELS_REMAINING:-39}"
  --dnum-values ${FULL_DNUM_VALUES:-1 2 3 4 5 6 7}
  --default-dnum "${FULL_DEFAULT_DNUM:-3}"
  --rotate-steps ${FULL_ROTATE_STEPS:--1}
  --warmup-cheap "${FULL_WARMUP_CHEAP:-30}"
  --timed-cheap "${FULL_TIMED_CHEAP:-100}"
  --warmup-medium "${FULL_WARMUP_MEDIUM:-20}"
  --timed-medium "${FULL_TIMED_MEDIUM:-50}"
  --timing-modes ${FULL_TIMING_MODES:-isolated batched}
  --batched-group-size "${FULL_BATCHED_GROUP_SIZE:-30}"
  --csv-timing-mode "${FULL_CSV_TIMING_MODE:-isolated}"
  --bootstrap-output-dir "$BOOTSTRAP_OUTPUT_DIR"
  --bootstrap-context-max-levels-remaining "${BOOTSTRAP_CONTEXT_MAX_LEVELS_REMAINING:-25}"
  --bootstrap-target-limbs-min "${BOOTSTRAP_TARGET_LIMBS_MIN:-1}"
  --bootstrap-dnum-values ${BOOTSTRAP_DNUM_VALUES:-2 3 4 5 6 7}
  --bootstrap-logbs-values ${BOOTSTRAP_LOGBS_VALUES:-12 13 14}
  --bootstrap-level-budgets ${BOOTSTRAP_LEVEL_BUDGETS:-3,3 4,4}
  --bootstrap-warmup-heavy "${BOOTSTRAP_WARMUP_HEAVY:-1}"
  --bootstrap-timed-heavy "${BOOTSTRAP_TIMED_HEAVY:-3}"
  --bootstrap-logN "${BOOTSTRAP_LOGN:-16}"
  --bootstrap-dcrt-bits "${BOOTSTRAP_DCRT_BITS:-52}"
  --bootstrap-first-mod "${BOOTSTRAP_FIRST_MOD:-55}"
  --bootstrap-secret-key-dist "${BOOTSTRAP_SECRET_KEY_DIST:-SPARSE_TERNARY}"
  --bootstrap-rescale-tech "${BOOTSTRAP_RESCALE_TECH:-FIXEDMANUAL}"
  --batched-report-md "$BATCHED_REPORT_MD"
  --batched-basic-csv-out "$BATCHED_BASIC_CSV_OUT"
  --batched-csv-out "$BATCHED_CSV_OUT"
)

if [[ -n "${BOOTSTRAP_TARGET_LIMBS_MAX:-}" ]]; then
  cmd+=(--bootstrap-target-limbs-max "$BOOTSTRAP_TARGET_LIMBS_MAX")
fi

if [[ "${SKIP_BATCHED_EXPORT:-0}" == "1" ]]; then
  cmd+=(--skip-batched-export)
fi

if [[ "${KEEP_CONTEXT_CACHE:-0}" == "1" ]]; then
  cmd+=(--keep-context-cache)
fi

if [[ "${NO_RESUME:-0}" == "1" ]]; then
  cmd+=(--no-resume)
fi

if [[ "${RERUN_FAILED:-0}" == "1" ]]; then
  cmd+=(--rerun-failed)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

printf '[run_all_profiling]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
