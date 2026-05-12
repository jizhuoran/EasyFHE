# ResNet20 AESPA

This application is the new NPZ-backed ResNet20 AESPA path.

It intentionally does not use the legacy bin/pkl weight readers from
`examples/resnet/src/utils.py`. All packed numeric vectors come from:

```text
examples/resnet/resnet20_aespa_weights.npz
```

Run from the repository root:

```bash
python -m examples.resnet20_aespa.main
```

Override the weight artifact with:

```bash
EASYFHE_RESNET20_AESPA_WEIGHTS=/path/to/resnet20_aespa_weights.npz \
python -m examples.resnet20_aespa.main
```
