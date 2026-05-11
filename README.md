# EasyFHE

EasyFHE is a tensor computation framework for Fully Homomorphic Encryption
(FHE). It is forked from PyTorch, but its goal is different: EasyFHE keeps the
general tensor runtime and adds encrypted tensor operators, while removing
neural-network-specific infrastructure that is not needed for FHE workloads.

The project is designed for writing FHE programs in a familiar Python tensor
style. Users describe computations over tensors, and EasyFHE handles the
cryptographic representation, homomorphic operators, CUDA kernels, key
switching, rescaling, rotations, and bootstrapping machinery underneath.

## What EasyFHE Provides

- **Tensor-style FHE programming**
  Use Python tensor code to express encrypted computation without manually
  managing ciphertext layout in every program.

- **GPU-accelerated homomorphic operators**
  EasyFHE includes CUDA implementations for core FHE primitives such as NTT,
  encoding, modulus switching, automorphism, ciphertext arithmetic, and fused
  homomorphic kernels.

- **OpenFHE-inspired CKKS behavior**
  The FHE layer follows the structure of mature CKKS implementations while
  adapting the execution path to a GPU tensor runtime.

- **Multi-GPU and NCCL-oriented runtime work**
  The repository keeps selected distributed and NCCL infrastructure for future
  multi-GPU FHE execution, while stripping distributed training algorithms that
  are not relevant to EasyFHE.

- **Lean PyTorch-derived core**
  EasyFHE retains the tensor, dispatch, CUDA, storage, serialization, and Python
  binding infrastructure needed by encrypted tensor workloads. Large PyTorch
  subsystems such as neural network modules, optimizers, TorchScript,
  functorch, ONNX/export, quantization, mobile, and most upstream test/CI/docs
  assets are being removed.

## Status

EasyFHE is an alpha research system. APIs and build options may change quickly.
The current repository is actively being reduced from a full PyTorch fork into
an FHE-focused tensor runtime.

Do not treat this project as a production cryptographic library yet. The FHE
scheme is intended to protect encrypted data, but the implementation is still
under active development and should be reviewed carefully before security
critical use.

## Examples

Example programs live under `examples/`. Current and historical examples
include:

- dot product
- bootstrapping
- logistic regression
- ResNet-20
- BERT Tiny
- sorting
- LSTM

A common smoke test is:

```bash
python3 examples/resnet/src/resnet20_aespa.py
```

## Building From Source

EasyFHE currently builds as an editable Python package. A typical CUDA/NCCL
development build is:

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

For a fully clean build, remove the build directory first:

```bash
rm -rf build
```

Then rerun the build command above.

## Repository Direction

EasyFHE is intentionally not a drop-in replacement for full PyTorch. The long
term target is:

- keep tensor storage, dispatch, CUDA runtime, selected autograd/runtime
  infrastructure, profiling hooks, distributed/NCCL primitives, and FHE ops
- remove neural network layers, optimizers, training-specific distributed
  algorithms, model export/compile stacks, mobile runtimes, quantization, and
  other general PyTorch features that do not serve FHE tensor execution

This makes the codebase smaller, easier to audit, and better aligned with FHE
systems work.

## Project Team

EasyFHE is primarily developed by contributors from Shandong University.
Maintainers and contributors include:

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
