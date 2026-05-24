# Bootstrap examples

This directory contains standalone OpenFHE bootstrapping checks and timing scripts used by the ResNet20 AESPA workflow.

## Scripts

- `benchmark_bootstrap.py`: benchmark bootstrap latency in isolation.
- `check_bootstrap_correctness.py`: decrypt-and-compare correctness check for bootstrap output.

## Run

From repository root:

```bash
python -m examples.bootstrap.benchmark_bootstrap --device cuda --warmup 1 --iters 3
python -m examples.bootstrap.check_bootstrap_correctness --device cuda --trace-stages
```

Use `--baby-step 8` or `--baby-step 8:8` to override the C2S/S2C BSGS baby-step count.

Both scripts reuse ResNet20 AESPA context defaults from `examples.resnet20_aespa.main`.
