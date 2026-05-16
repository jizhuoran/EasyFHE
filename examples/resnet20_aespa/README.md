# ResNet20 AESPA

This application is the self-contained NPZ-backed ResNet20 AESPA path.

It intentionally does not use the legacy bin/pkl weight readers or data helpers
from `examples/resnet`. The default input data and packed numeric vectors live
inside this directory:

```text
examples/resnet20_aespa/data/cifar10/test_batch.bin
examples/resnet20_aespa/resnet20_aespa_weights.npz
```

Run from the repository root:

```bash
python -m examples.resnet20_aespa.main
```

The default device is CUDA. To run the same path on CPU:

```bash
python -m examples.resnet20_aespa.main --device cpu
```

Override the weight artifact with:

```bash
EASYFHE_RESNET20_AESPA_WEIGHTS=/path/to/resnet20_aespa_weights.npz \
python -m examples.resnet20_aespa.main
```

Override the CIFAR-10 test batch with either:

```bash
EASYFHE_CIFAR10_TEST_BATCH=/path/to/test_batch.bin \
python -m examples.resnet20_aespa.main
```

or:

```bash
EASYFHE_RESNET20_AESPA_DATA_DIR=/path/to/data \
python -m examples.resnet20_aespa.main
```

where the second form expects `/path/to/data/cifar10/test_batch.bin`.
