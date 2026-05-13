# EasyFHE

EasyFHE is a tensor runtime for Fully Homomorphic Encryption (FHE). It began as
a PyTorch fork, but it is being narrowed into something more specific: a compact
runtime for encrypted tensor programs, CKKS-style homomorphic operators, CUDA
kernels, native key material generation, and FHE-oriented examples.

```python
import easyfhe
import easyfhe.fhe as fhe
```

EasyFHE keeps the tensor infrastructure that encrypted workloads need and
removes large parts of the training stack that do not serve FHE execution.

## Highlights

- `easyfhe.fhe` frontend for CKKS context specs, ciphertext state, plaintext
  preparation, arithmetic, rotation, rescale, fused ops, and bootstrapping.
- Native sampler and material generation for CKKS keys and rotation keys.
- CUDA accelerated FHE kernels for encoding, NTT, automorphism, key switching,
  modulus movement, ciphertext arithmetic, and bootstrapping paths.
- A PyTorch-derived tensor core with storage, dispatch, CUDA runtime, Python
  bindings, selected profiling, and selected NCCL-oriented infrastructure.
- Self-contained research examples and benchmarking tools under `examples/`.

## Install

Prebuilt wheels are being prepared for Linux x86_64, Python 3.12, and CUDA
runtime variants. The intended public install shape is PyTorch-like: keep the
package name as `easyfhe`, then choose a CUDA wheel channel.

CUDA 13.2:

```bash
python -m pip install "easyfhe==2.13.0a0+cu132.git6e869e1" \
  --find-links https://jizhuoran.github.io/EasyFHE/whl/cu132
```

CUDA 12.4:

```bash
python -m pip install "easyfhe==2.13.0a0+cu124.git6e869e1" \
  --find-links https://jizhuoran.github.io/EasyFHE/whl/cu124
```

The wheel links become usable after the corresponding GitHub Release assets are
uploaded. Until then, use the source build path below.

## Build From Source

A typical local CUDA/NCCL editable build is:

```bash
python3 setup.py clean

USE_EASYFHE_FAST_BUILD=1 \
USE_EASYFHE_FAST_INFERENCE=1 \
USE_CUDA=1 \
USE_DISTRIBUTED=1 \
USE_NCCL=1 \
USE_GLOO=0 \
USE_TENSORPIPE=0 \
USE_MPI=0 \
USE_UCC=0 \
USE_CUDNN=0 \
USE_MKLDNN=0 \
BUILD_TEST=0 \
BUILD_FUNCTORCH=0 \
USE_NINJA=1 \
CMAKE_BUILD_TYPE=Release \
TORCH_CUDA_ARCH_LIST=8.0 \
MAX_JOBS=24 \
python3 setup.py develop
```

For a fully clean rebuild:

```bash
rm -rf build
```

Then rerun the build command.

## Examples

Self-contained ResNet-20 AESPA example:

```bash
python3 examples/resnet20_aespa/main.py
```

Legacy ResNet-20 AESPA smoke program:

```bash
python3 examples/resnet/src/resnet20_aespa.py
```

Dot product example:

```bash
python3 examples/dot_product/innerproduct_example.py
```

Benchmark harness:

```bash
python3 examples/easyfhe_benchmark/cli.py --help
```

## Repository Map

- `easyfhe/`: Python package and retained tensor runtime surface.
- `easyfhe/fhe/`: FHE frontend, context generation, ops, bootstrapping, runtime
  options, and material handling.
- `aten/src/ATen/native/fhe/`: native FHE kernels and native sampler.
- `examples/resnet20_aespa/`: self-contained AESPA ResNet-20 example.
- `examples/easyfhe_benchmark/`: profiling and benchmark harness.
- `packaging/`: manylinux wheel and container packaging scripts.
- `index.html` and `styles.css`: static project website for GitHub Pages.

## Website

The project website lives at the root of the `docs` branch so GitHub Pages can
serve it directly.

Local preview:

```bash
python3 -m http.server 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

For GitHub Pages, set:

```text
Source: Deploy from a branch
Branch: docs
Folder: /
```

The public page will be:

```text
https://jizhuoran.github.io/EasyFHE/
```

## Current Status

EasyFHE is an alpha research system. APIs, build flags, ciphertext layout, and
packaging details may change quickly. Do not treat it as a production
cryptographic library without independent review.

## Direction

EasyFHE is not trying to remain a full PyTorch clone. The long-term direction is
to keep tensor storage, dispatch, CUDA execution, selected profiling, selected
NCCL runtime pieces, and FHE operators, while removing model-training,
optimizer, export, mobile, quantization, and other general ML surfaces.

## Contributors

EasyFHE is primarily developed by contributors from Shandong University.

- [Zhuoran Ji](https://github.com/jizhuoran)
- [Honghui You](https://github.com/youhonghui)
- [Wenzhe Wang](https://github.com/Kelly-Zhe)
- [Haoping Yang](https://github.com/er1ciac)
- [Kanyu Ye](https://github.com/kanyuYe)
- [Yuhang Fan](https://github.com/Azathoth13)
- [Yusi Chen](https://github.com/chenyusii)

## License

EasyFHE is distributed under the GPL-3.0 license.

EasyFHE is derived from PyTorch and keeps portions of the PyTorch runtime. See
the repository license files and retained third-party notices for upstream
license information.
