# EasyFHE Benchmark

This directory contains the EasyFHE latency benchmark pipeline. It runs real
EasyFHE operations, writes JSON timing artifacts, and exports simulator-facing
CSV tables.

## Entry Points

Run from the repository root:

```bash
python examples/easyfhe_benchmark/cli.py --preset full-pipeline --dry-run
```

Compatibility wrapper:

```bash
bash examples/easyfhe_benchmark/run_all_profiling.sh
```

Available presets:

- `bootstrap-profiling`
- `full-profiling`
- `full-pipeline`

## What It Does

- `timed_harness.py` measures foundational ops directly. Supported tuples
  produce timing rows; locally invalid tuples are reported as `unsupported`;
  exceptions are reported as `failed`.
- `bootstrap-profiling` is a thin CLI alias for the bucketed bootstrap runner.
- `run_bootstrap_bucketed_profiling.py` profiles bootstrap by reusing one
  context per `(dnum, slots, level_budget)` bucket.
- `tools/export_basic_ops_report_and_csv.py` exports foundational op timings.
- `tools/update_bootstrap_rows_in_csv.py` merges bootstrap timings into the CSV.

The exported CSV schema is:

```text
op, accelerator, N, cur_limbs, target_limbs, dnum, slots, level_budget_0, level_budget_1, latency_us
```

## Execution Order

`full-pipeline` runs:

1. `full-profiling` for foundational ops.
2. foundational CSV export.
3. bucketed bootstrap profiling.
4. optional batched foundational export and bootstrap merge.

Foundational ops currently run op-major. Bootstrap runs bucket-major:

```text
for dnum:
  for logbs:
    for level_budget:
      load one context/key bucket
      for target_limbs:
        profile the pending cases
```

## Defaults

Output files are written under:

```text
examples/easyfhe_benchmark/data/
```

Context cache defaults to:

```text
examples/easyfhe_benchmark/data/context_store/
```

The full pipeline uses `CONTEXT_CACHE_MODE=transient` by default in the shell
wrapper, so generated contexts are reused within a run and then cleaned unless
`KEEP_CONTEXT_CACHE=1` is set.

Profiling is resumable by default. Each timed JSON contains a stable
`case_manifest`, a `case_id` for every result, and resume metadata. If a run is
interrupted, rerun the same command and completed `ok`, `unsupported`, and
`failed` cases are skipped. Use `--rerun-failed` to retry failed cases, or
`--no-resume` to ignore existing outputs and measure everything again.

Bucketed bootstrap profiling also writes:

```text
examples/easyfhe_benchmark/data/bootstrap_bucketed_fullscan/case_manifest.json
```

## Useful Overrides

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
DEVICE=cuda \
CONTEXT_CACHE_MODE=transient \
bash examples/easyfhe_benchmark/run_all_profiling.sh
```

Small dry run:

```bash
python examples/easyfhe_benchmark/cli.py --preset full-pipeline \
  --dry-run \
  --device cpu \
  --limb-min 1 --limb-max 1 \
  --bootstrap-dnum-values 2 \
  --bootstrap-logbs-values 12 \
  --bootstrap-level-budgets 3,3 \
  --bootstrap-target-limbs-max 17
```

## Generated Artifacts

Most files under `data/` are generated benchmark outputs and should not be
checked in.
