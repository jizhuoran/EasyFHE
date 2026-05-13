# EasyFHE

EasyFHE is a tensor computation runtime for Fully Homomorphic Encryption
(FHE). It started as a PyTorch fork, but it is not trying to be a full PyTorch
replacement. The goal is narrower: keep the tensor runtime pieces that FHE
programs need, remove neural-network-specific machinery, and expose encrypted
tensor operations through a familiar Python interface.

```python
import easyfhe as torch
import easyfhe.fhe as fhe
```

## Why EasyFHE Exists

FHE systems need high-performance tensor infrastructure, CUDA kernels, memory
management, dispatch, serialization, profiling, and eventually multi-GPU
communication. They do not need most of the training stack: neural network
layers, optimizers, TorchScript, ONNX export, mobile runtimes, quantization, or
large upstream test and CI surfaces.

EasyFHE trims the runtime toward encrypted tensor execution.

## What Is Included

- CKKS-oriented FHE frontend under `easyfhe.fhe`
- Native context generation for keys, ciphertext material, and rotation keys
- CUDA accelerated FHE kernels for encoding, NTT, automorphism, key switching,
  modulus movement, ciphertext arithmetic, fused kernels, and bootstrapping
- Tensor storage, dispatch, CUDA runtime, Python bindings, and selected
  profiling/runtime infrastructure inherited from PyTorch
- Selected distributed/NCCL primitives for future multi-GPU FHE work
- Research examples and smoke tests under `examples/`

## Current Status

EasyFHE is an alpha research system. APIs, build flags, and ciphertext layout
may change quickly. It should not be treated as a production cryptographic
library without independent review.

The repository has already been heavily reduced from upstream PyTorch. The
remaining code is being shaped around FHE tensor execution rather than general
machine learning.

## Quick Build

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

## Smoke Test

```bash
python3 examples/resnet/src/resnet20_aespa.py
```

The self-contained ResNet-20 AESPA example lives in:

```bash
examples/resnet20_aespa/
```

## Repository Map

- `easyfhe/`: Python package and retained tensor runtime surface
- `easyfhe/fhe/`: FHE frontend, context generation, ops, bootstrapping, runtime
  options, and material handling
- `aten/src/ATen/native/fhe/`: native FHE kernels and native sampler
- `examples/`: research examples, benchmarks, and smoke programs
- `packaging/`: optional wheel/container packaging scripts
- `third_party/`: retained third-party dependencies required by the runtime

## Website

A small static website is in `website/`. Open `website/index.html` directly in
a browser, or serve it with any static file server.

## Project Direction

EasyFHE should become a compact tensor runtime for encrypted computation:

- keep tensor infrastructure, CUDA execution, selected profiling, selected
  distributed/NCCL primitives, and the FHE layer
- remove training-specific and model-authoring features that do not serve FHE
  programs
- make the Python surface `easyfhe` first, while keeping only the compatibility
  shims needed by the current runtime

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
