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
- `USE_EASYFHE_CPU_AVX=OFF`
- `BUILD_FUNCTORCH=OFF`
- `BUILD_LAZY_CUDA_LINALG=OFF`

From `build/compile_commands.json`:

NCCL note: the fast build defaults to `USE_NCCL=ON` and `USE_C10D_NCCL=ON`
unless `USE_NCCL=0` is set explicitly. After changing the CMake cache, run the
incremental develop step so `easyfhe/lib/libtorch_python.so` is refreshed; a
plain `torch_python` target build leaves Python importing the previous installed
library.

- A fresh build from an empty `build/` confirms most pruning already happens
  through the EasyFHE fast-build source filters in `caffe2/CMakeLists.txt`.
- `aten/src/ATen/native/fhe/**` is actively compiled. Keep.
- `aten/src/ATen/native/cuda/*.cu` is heavily compiled. Treat as tensor/CUDA infrastructure.
- `aten/src/ATen/native/cuda/SpectralOps.cpp` is still compiled even though the public Python `easyfhe.fft` package was removed.
- `aten/src/ATen/native/easyfhe_quantized_stubs.cpp` is still compiled. It backs disabled quantized type/link symbols, not the public quantization API.
- `aten/src/ATen/functorch/easyfhe_functorch_stubs.cpp` is still compiled. It backs disabled functorch symbols, not the public `_functorch` Python package.
- Several `aten/src/ATen/native/cpu/*.cpp` files are compiled through generated
  `build/aten/src/ATen/native/cpu/*.cpp.{DEFAULT,AVX2,AVX512}.cpp` wrappers.
  Do not delete CPU kernel sources based on stale direct-path compile database
  checks.
- The EasyFHE fast build now defaults `USE_EASYFHE_CPU_AVX=OFF`, so those CPU
  wrappers should only be generated as `.DEFAULT.cpp`; SLEEF AVX2 and AVX512F
  dispatcher targets are disabled in the same profile. SLEEF's scalar purecfma
  fallback still compiles with `-mavx2 -mfma`; attempting to remove only
  `-mavx2` fails on GCC because that SLEEF mode requires `FP_FAST_FMA/FMAF`.
  Re-enable with `USE_EASYFHE_CPU_AVX=1` if a later CPU performance experiment
  needs ATen CPU capability wrappers.
- `aten/src/ATen/native/cuda/linalg/**` was deleted after a clean build proved it
  is not needed by the fast build.
- `aten/src/ATen/native/kleidiai/**` was deleted after a clean build proved it is
  not needed by the fast build.
- `aten/src/ATen/native/sparse/**` is not compiled as source files in the current build, but some headers are included by compiled sparse/layout utility code.
- `aten/src/ATen/quantized/**` headers are not compiled directly, but are included by quantized stubs and core tensor shape/type code.

Approximate tracked-source sizes for useful deletion accounting:

| Path | Size | Current build status | Notes |
| --- | ---: | --- | --- |
| `aten/src/ATen/native/cpu/**` | 716 KiB | selected `.cpp` files compile through generated CPU-capability wrappers | Keep for now; fresh build disproved the earlier source-only deletion idea. |
| `aten/src/ATen/native/cuda/**` | 1.6 MiB | 33 compiled, 77 not compiled | Do this by op family, not wholesale. |
| `aten/src/ATen/cpu/vec/**` | 1.5 MiB | headers only | Dense CPU/vector utility headers; keep until CPU support is intentionally dropped. |
| `aten/src/ATen/native/cuda/linalg/**` | 68 KiB | removed | Deleted in the first native-source cleanup batch. |
| `aten/src/ATen/native/kleidiai/**` | 44 KiB | removed | Deleted in the first native-source cleanup batch. |
| `aten/src/ATen/native/sparse/**` | 32 KiB | headers only | Keep until sparse type/layout cleanup. |
| `aten/src/ATen/templates/**` | 200 KiB | codegen templates | Do not delete by hand; codegen input. |

## Removed Native Sources

| Path | Why | Verification |
| --- | --- | --- |
| `aten/src/ATen/native/cuda/linalg/**` | Public `easyfhe.linalg` had already been removed, fast build had `BUILD_LAZY_CUDA_LINALG=OFF`, and fresh configure/build did not require these sources. | Removed after `rm -rf build`; full `tools/easyfhe_fast_build.sh` passed, import/NumPy/NCCL/profiler smoke passed, and CUDA ResNet20 AESPA passed. |
| `aten/src/ATen/native/kleidiai/**` | Public `easyfhe.backends.kleidiai` had already been removed and current fast build does not compile the KleidiAI CPU int4 path. | Same clean-build and runtime smoke as above. Note: `aten/src/ATen/native/cpu/int4mm_kernel.cpp` still contains a stale include and should be handled with the later CPU/int4 cleanup, not by restoring KleidiAI. |

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
| `aten/src/ATen/native/cpu/{CatKernel.cpp,StackKernel.cpp,SumKernel.cpp,UnaryOpsKernel.cpp,BinaryOpsKernel.cpp,CopyKernel.cpp,ReduceOpsKernel.cpp,...}` and related headers | Keep for now: fresh builds compile selected CPU kernels through generated CPU-capability wrappers. |

## Good First Native Deletion Candidates

These are source directories/files that are either not compiled in the current
build or match Python surfaces already removed.

| Candidate | Why | Test requirement |
| --- | --- | --- |
| Uncompiled special/math CUDA families already filtered by fast build, e.g. `aten/src/ATen/native/cuda/*bessel*`, `*chebyshev*`, `UnaryGeometric*`, `UnaryGamma*`, `UnaryLog*`, `ZetaKernel.cu` | Public `easyfhe.special` was removed and current fast build excludes these files. | Delete by family in small batches. Rebuild `torch_cuda`; check no generated registration references force missing symbols. |
| `aten/src/ATen/native/sparse/cuda/cuSPARSELtOps.h` plus `torch/csrc/cuda/shared/cusparselt.cpp` follow-up | Python `backends/cusparselt` was removed and `USE_CUSPARSELT=OFF`. | Check include references first; delete only with CMake/build validation. |
| Generated Python binding remnants for public `fft/linalg/special/sparse/nested/masked/quantized` modules | Python packages were removed or trimmed to anchors. | Prefer generator/config changes over deleting generated files by hand. Run codegen + build. |

## Source-Deletion Batches To Try Next

These batches are ordered to keep rollback small and to avoid touching Tensor
core/type infrastructure too early.

1. **Special/math CUDA source-only pass**
   - Delete filtered CUDA special/math families by small groups:
     bessel/chebyshev/hermite/laguerre/legendre/modified_bessel,
     `UnaryGeometric*`, `UnaryGamma*`, `UnaryLog*`, `ZetaKernel.cu`,
     `IGammaKernel.cu`, `GcdLcmKernel.cu`.
   - Keep currently compiled CUDA tensor basics such as `AbsKernel.cu`,
     `UnaryOpsKernel.cu`, `UnarySignKernels.cu`, `UnaryComplexKernels.cu`,
     `TensorCompare.cu`, `Compare*.cu`, `Binary*` files that are in
     `compile_commands.json`.
2. **cuSPARSE/cuSPARSELt crumbs**
   - Delete uncompiled `aten/src/ATen/cuda/CUDASparse*.cpp`,
     `aten/src/ATen/cuda/CuSparseHandlePool.cpp`, and the cuSPARSELt Python/C++
     wrapper only after checking no header include survives in compiled code.
   - Keep `cusparse` link trimming separate because CUDA dependency setup
     currently adds it broadly.
3. **Link-dependency cleanup**
   - `libtorch_cuda.so` still reports unused direct dependencies on `cusolver`,
     `cusparse`, `cufft`, and `curand`. Trim these through CUDA dependency setup
     in separate commits after source deletion has stabilized.
4. **CPU/int4 cleanup, later**
   - Do not delete `aten/src/ATen/native/cpu/*.cpp` wholesale. Fresh builds
     compile selected CPU kernels through generated wrappers.
   - The stale `int4mm_kernel.cpp` -> KleidiAI include is a focused cleanup
     candidate, but it should be handled with generated op/schema checks.
5. **Type-system cleanup, later**
   - Sparse, nested, quantized, functorch, functionalization, and FFT require
     native schema/type/codegen cleanup before source removal. These are not
     first-pass source deletions.

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

1. Special/math CUDA source-only cleanup:
   - delete uncompiled CUDA special/math kernel families that match removed public `special` surface
2. cuSPARSE/cuSPARSELt crumbs:
   - inspect and delete uncompiled sparse CUDA helper sources/wrappers, then trim link dependencies separately
3. FFT native cleanup:
   - remove `SpectralOps.cpp` by pruning FFT op schemas/generated bindings first
   - remove `cufft` link dependency if no remaining user
4. Quantized type cleanup:
   - remove public/generated quantized ops first
   - then simplify `QTensorImpl`/`Quantizer` core references
   - only then remove `easyfhe_quantized_stubs.*`
5. Functorch/native transform cleanup:
   - remove remaining `TensorWrapper` and disabled native functorch stubs after C++ references are gone
6. Nested/sparse type cleanup:
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
