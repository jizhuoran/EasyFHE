This repo contains source code for HELR benchmark. 

Prerequisite:
- GPU-FHE
- OpenFHE--catslab version

1. Run the project

Run:
```bash
cd examples/helr/src/baseline
python3 ./helr_GPU.py
```

The computation flow of `examples/helr/src/baseline/helr_GPU.py` closely follows the method presented in `Logistic Regression on Homomorphic Encrypted Data at Scale (AAAI 2019)`, with adaptations for transitioning from a multi-precision implementation to an RNS-based approach.
Its original repo is https://github.com/KyoohyungHan/HELR.