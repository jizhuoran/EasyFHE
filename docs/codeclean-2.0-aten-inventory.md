# codeclean-2.0 ATen native cleanup inventory

This note maps the current `aten/` native tree after the Python surface cleanup.
Unlike `easyfhe/`, this tree is build/codegen/native-dispatch infrastructure.
Do not delete large subtrees without a build/link check.

## Current Build Facts

From `build/CMakeCache.txt`:

- `USE_CUDA=1`
- `USE_DISTRIBUTED=ON`
- `USE_NCCL=ON`
- `USE_C10D_NCCL=ON`
- `USE_CUDNN=OFF`
- `USE_CUSPARSELT=OFF`
- `USE_MKLDNN=OFF`
- `USE_MPS=OFF`
- `USE_ROCM=0`
- `USE_MSLK=OFF`
- `BUILD_FUNCTORCH=OFF`
- `BUILD_LAZY_CUDA_LINALG=OFF`

From `build/compile_commands.json`:

NCCL note: the fast build defaults to `USE_NCCL=ON` and `USE_C10D_NCCL=ON`
unless `USE_NCCL=0` is set explicitly. After changing the CMake cache, run the
incremental develop step so `easyfhe/lib/libtorch_python.so` is refreshed; a
plain `torch_python` target build leaves Python importing the previous installed
library.

- `aten/src/ATen/native/fhe/**` is actively compiled. Keep.
- `aten/src/ATen/native/cuda/*.cu` is heavily compiled. Treat as tensor/CUDA infrastructure.
- `aten/src/ATen/native/cuda/SpectralOps.cpp` is still compiled even though the public Python `easyfhe.fft` package was removed.
- `aten/src/ATen/native/easyfhe_quantized_stubs.cpp` is still compiled. It backs disabled quantized type/link symbols, not the public quantization API.
- `aten/src/ATen/functorch/easyfhe_functorch_stubs.cpp` is still compiled. It backs disabled functorch symbols, not the public `_functorch` Python package.
- `aten/src/ATen/native/cuda/linalg/**` is not compiled in the current build.
- `aten/src/ATen/native/kleidiai/**` is not compiled in the current build.
- `aten/src/ATen/native/sparse/**` is not compiled as source files in the current build, but some headers are included by compiled sparse/layout utility code.
- `aten/src/ATen/quantized/**` headers are not compiled directly, but are included by quantized stubs and core tensor shape/type code.

## Keep

| Path | Reason |
| --- | --- |
| `aten/src/ATen/native/fhe/**` | EasyFHE native CPU/CUDA kernels and sampler. |
| `aten/src/ATen/native/cuda/*.cu` selected by current build | CUDA tensor runtime used by FHE and tensor materialization. |
| `aten/src/ATen/cuda/**` | CUDA context, streams, graphs, allocator, BLAS handles, and runtime support. |
| `aten/src/ATen/core/**` | Dispatcher, TensorImpl, IValue, schema/type system, storage, and generated op plumbing. |
| `aten/src/ATen/native/{TensorShape,TensorFactories,Copy,Resize,UnaryOps,BinaryOps,ReduceOps,TensorCompare,TensorAdvancedIndexing}.cpp` and similar core files | Dense tensor runtime. Keep unless an op-level generated build proves a file is unused. |
| `aten/src/ATen/native/easyfhe_quantized_stubs.*` | Keep for now: compiled stubs satisfy quantized Tensor/Quantizer symbols used by core headers and tensor shape code after public quantization was removed. |
| `aten/src/ATen/functorch/easyfhe_functorch_stubs.cpp` | Keep for now: compiled disabled symbol shim after Python `_functorch` was removed. |
| `aten/src/ATen/quantized/{QTensorImpl.h,Quantizer.h}` | Keep for now because compiled core code still includes quantizer types. Remove only after quantized tensor type is removed from core `TensorImpl`/`IValue`/schema paths. |

## Good First Native Deletion Candidates

These are source directories/files that are either not compiled in the current
build or match Python surfaces already removed.

| Candidate | Why | Test requirement |
| --- | --- | --- |
| `aten/src/ATen/native/cuda/linalg/**` | Public `easyfhe.linalg` is deleted; current compile database shows zero files from this directory. | Delete as one batch, reconfigure if needed, rebuild `torch_cuda`, run import + GPU ResNet. Also remove unnecessary `cusolver` dependency if link still carries it. |
| `aten/src/ATen/native/kleidiai/**` | Python `backends/kleidiai` was removed; current build has `USE_MSLK=OFF` and zero compiled files from this directory. | Delete as one batch, rerun CMake configure/build smoke. |
| `aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.h` plus `torch/csrc/cuda/shared/cusparselt.cpp` follow-up | Python `backends/cusparselt` was removed and `USE_CUSPARSELT=OFF`. | Check include references first; delete only with CMake/build validation. |
| Generated Python binding remnants for public `fft/linalg/special/sparse/nested/masked/quantized` modules | Python packages were removed or trimmed to anchors. | Prefer generator/config changes over deleting generated files by hand. Run codegen + build. |

## Medium Risk Candidates

| Candidate | Why risky |
| --- | --- |
| `aten/src/ATen/native/cuda/SpectralOps.cpp` | Public FFT is gone, but this file is still compiled and `cufft` is linked. Removing it needs op/schema/codegen cleanup for FFT ops and CMake dependency trimming. |
| `aten/src/ATen/native/sparse/*.h` | Sparse Python API is trimmed, but native sparse tensor class installation still requires sparse layout/type concepts. Headers are included by compiled code such as sparse tensor utility paths. |
| `aten/src/ATen/NestedTensorImpl.cpp` and nested generated registrations | Public nested Python package is gone, but nested type/layout concepts still appear in compiled core/generated code. Needs schema/type-system cleanup, not source deletion first. |
| MPS/MTIA/XPU hook interface files under `aten/src/ATen/detail/` | Backend Python packages were removed, but generic hook interfaces may still compile into core. Remove only after `Context`/hook lookup paths are simplified. |
| `aten/src/ATen/native/easyfhe_quantized_stubs.*` | Looks tempting because quantization is gone, but it is actively compiled to satisfy quantizer/QTensor symbols. Remove only after quantized core type erasure. |
| `aten/src/ATen/functorch/easyfhe_functorch_stubs.cpp` | Looks tempting because `_functorch` Python is gone, but it is actively compiled as a disabled symbol shim. Remove only after all native references to functorch/TensorWrapper are gone. |

## High Risk / Do Not Start Here

| Area | Why |
| --- | --- |
| `aten/src/ATen/core/**` | Dispatcher, schema, TensorImpl, IValue, generated operator plumbing. |
| `aten/src/ATen/native/*.cpp` wholesale | Many files are dense tensor runtime even when public PyTorch namespaces are removed. |
| `aten/src/ATen/native/cuda/*.cu` wholesale | Current FHE path still depends on CUDA tensor factories, copy, indexing, reductions, unary/binary kernels, and CUDA storage/materialization. |
| `aten/src/ATen/templates/**` and generated `aten/src/ATen/ops/**` | Codegen inputs/outputs. Remove through torchgen/native_functions pruning, not ad hoc deletion. |

## Suggested Deletion Batches

1. Inactive native backend crumbs:
   - `aten/src/ATen/native/cuda/linalg/**`
   - `aten/src/ATen/native/kleidiai/**`
   - possible cuSPARSELt header/source follow-up
2. FFT native cleanup:
   - remove `SpectralOps.cpp` by pruning FFT op schemas/generated bindings first
   - remove `cufft` link dependency if no remaining user
3. Quantized type cleanup:
   - remove public/generated quantized ops first
   - then simplify `QTensorImpl`/`Quantizer` core references
   - only then remove `easyfhe_quantized_stubs.*`
4. Functorch/native transform cleanup:
   - remove remaining `TensorWrapper` and disabled native functorch stubs after C++ references are gone
5. Nested/sparse type cleanup:
   - keep sparse anchors until native sparse tensor class installation is removed or replaced
   - remove nested generated registration only after core type references are absent

Minimum validation for each native batch:

```bash
cmake --build build --target torch_cpu torch_cuda -j$(nproc)
python3 -X faulthandler -c "import easyfhe; print(easyfhe.__version__); print(easyfhe.tensor([1.0]))"
python3 -X faulthandler -c "import easyfhe.fhe as fhe; print(len(fhe.__all__))"
python3 -X faulthandler -c "import easyfhe.profiler; print('profiler ok')"
CUDA_VISIBLE_DEVICES=0 EASYFHE_DEVICE=cuda python3 -m examples.resnet20_aespa.main
```
