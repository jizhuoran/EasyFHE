# codeclean-2.0 easyfhe directory inventory

This is a top-level inventory for `easyfhe/`. It explains what each directory
appears to be, how it relates to the FHE runtime, and what cleanup posture makes
sense after the current Python-surface pruning.

Legend:

- Keep: needed for the current runtime or explicitly wanted.
- Keep narrow: keep for now, but it should eventually be narrowed.
- Keep minimal: keep only a tiny compatibility/native-initialization anchor.
- Candidate: likely removable, but test as a separate batch.
- Removed: already deleted in this branch.
- Build payload: installed headers/libs/share data; shrink through build rules.

## Current Core

| Directory | Posture | What it is | Notes |
| --- | --- | --- | --- |
| `fhe/` | Keep | EasyFHE public FHE runtime: CKKS context, ciphertext state, bootstrap, runtime CLI/options, material/key helpers, FHE ops. | This is the project center. |
| `cuda/` | Keep | CUDA runtime bindings, streams/events, memory, NCCL Python wrapper, CUDA device helpers. | Needed by GPU AESPA ResNet. Preserve `cuda/nccl.py` even while NCCL build support is fixed separately. |
| `cpu/` | Keep | CPU runtime helpers. | Keep unless CPU support is intentionally dropped. The deprecated `cpu/amp/` compatibility subpackage has been removed. |
| `autograd/` | Keep narrow | Autograd/profiler utilities and native autograd bindings. | FHE inference may not need full autograd, but native import and profiler still touch this area. |
| `_C/` | Keep | Python typing/stub package for native `_C` extension submodules. | Do not prune casually; native extension import depends on `_C`. |
| `_C_flatbuffer/` | Candidate | Flatbuffer typing/stub surface. | Likely tied to export/serialization/compiler features; test separately. |
| `profiler/` | Keep | Kineto/profiler API for runtime/performance traces. | User wants this. Optimizer-step hook was removed when `optim/` was deleted. |
| `distributed/` | Keep | ProcessGroup/c10d/distributed Python surface. | Keep because NCCL/broadcast/all-reduce primitives are wanted. Current build has `USE_NCCL=OFF`; enabling NCCL is separate. |
| `futures/` | Keep narrow | Future wrapper used by distributed/RPC-style APIs. | Keep while distributed is kept. |
| `multiprocessing/` | Keep narrow | PyTorch multiprocessing/reductions helpers. | Keep while serialization/distributed compatibility is unresolved. |

## Tensor Runtime Compatibility

| Directory | Posture | What it is | Notes |
| --- | --- | --- | --- |
| `_ops/` | Keep | Not a directory, but top-level `easyfhe._ops` is important: operator namespace and dispatch plumbing. | Mentioned here because many dirs depend on it. |
| `_library/` | Keep narrow | Custom op/library registration, fake impls, schema inference, Triton helpers. | Needed for some operator plumbing. Export-related paths inside can be narrowed later. |
| `_dispatch/` | Keep narrow | Python dispatch/TorchDispatch helper surface. | Shared tensor runtime plumbing. |
| `_prims/` | Keep narrow | Primitive operator definitions/context/debug/rng prims. | Runtime/compiler-adjacent; prune with op tests. |
| `_prims_common/` | Keep | Common primitive/type utilities. | Many tensor paths depend on it. |
| `_refs/` | Keep narrow | Reference implementations for ATen-style ops, including internal FFT/linalg/special refs. | Public `easyfhe.fft/linalg/special` are gone, but internal refs may still be used by decomposition/runtime code. |
| `_subclasses/` | Keep narrow | Fake/functional/complex tensor subclass support. | Often pulled in by dispatch/compiler paths; prune after compiler stack decisions. |
| `_numpy/` | Removed | NumPy-compatible namespace helpers. | Removed. Core `torch.from_numpy`, `Tensor.numpy()`, and `Tensor.__array__` are native tensor conversion paths and remain available. |
| `_native/` | Removed | Python helpers for native/Triton custom ops. | Removed after no project references were found and import failed through Triton/xpu compatibility paths. |
| `nested/` | Removed | NestedTensor compatibility. | Removed after import was found broken through missing native `_C._nested`. |
| `masked/` | Removed | MaskedTensor compatibility. | Removed after import was found broken through missing ATen masked-op registrations. |
| `sparse/` | Keep minimal | Native sparse tensor class anchor. | Python sparse API implementation was removed, but tiny `easyfhe.sparse` and `easyfhe.cuda.sparse` modules must remain because `_C._initExtension()` imports them while installing sparse tensor classes. Native sparse/layout concepts were not touched. |

## Compiler, Export, and Transform Stack

| Directory | Posture | What it is | Notes |
| --- | --- | --- | --- |
| `_dynamo/` | Removed | TorchDynamo-style config/decorator surface. | Removed with a top-level `easyfhe.compiler` object that returns non-compiling state for internal runtime checks. |
| `_inductor/` | Removed | Inductor/kernel template surface. | Removed; profiler/cudagraph config paths now tolerate the missing compiler backend. |
| `_export/` | Removed | Internal export implementation. | Removed after public `export/` was already deleted. |
| `compiler/` | Removed | Public compiler/cache/config surface. | Removed as an importable package; `easyfhe.compiler` remains a small in-memory disabled namespace for CUDA/profiler guards. |
| `fx/` | Keep narrow | FX graph tooling plus symbolic-shape helpers. | Do not delete wholesale: `SymInt/SymBool`, `_refs`, `_prims_common`, meta registrations, and tensor subclass helpers depend on `fx.experimental.symbolic_shapes`, `sym_node`, traceback, and schema utilities. Public tracing/graph pieces can be inspected later for narrowing. |
| `_decomp/` | Removed | Operator decomposition tables. | Removed after it was found to be unusable in the fast build path because importing it pulls FakeTensor/functorch native pieces that are absent. |
| `_functorch/` | Removed | Functorch/vmap/autograd-function compatibility. | Public `func/` is deleted; `_ops` keeps a disabled fallback for transform dispatch state. |
| `_higher_order_ops/` | Removed | Higher-order operator implementations/passes. | Compiler/functorch-related; small `new_token_tensor` and `autograd_not_implemented` helpers were inlined into `_prims`/`rng_prims`. |
| `_custom_op/` | Keep narrow | Custom op API compatibility. | Imports successfully and is closer to `torch.library`/operator registration infrastructure than graph compiler. Keep unless a later operator-infra pass proves it unused. |
| `_lazy/` | Removed | Lazy tensor backend compatibility. | Removed; not current FHE surface. |
| `_awaits/` | Removed | JIT awaitable helpers. | Removed; `_jit_internal` keeps a local placeholder class for retained compatibility code. |

## Device/Backend Surfaces

| Directory | Posture | What it is | Notes |
| --- | --- | --- | --- |
| `backends/` | Keep narrow | Backend feature flags and backend-specific helpers. | Trimmed to the backends still useful for the CUDA/FHE runtime: `cuda`, `cpu`, and `openmp`. |
| `accelerator/` | Keep narrow | Generic accelerator abstraction. | Runtime may use generic device helpers. |
| `amp/` | Removed | Autocast/GradScaler surface. | Training/mixed precision compatibility, not FHE-specific. Removed along with `cuda/amp/` and `cpu/amp/`. |
| `mps/` | Removed | Apple MPS compatibility. | Removed for Linux/CUDA-focused runtime. |
| `mtia/` | Removed | MTIA accelerator compatibility. | Removed. |
| `numa/` | Removed | NUMA binding helpers. | Removed. |
| `monitor/` | Keep | Native monitor/wait-counter surface with TensorBoard event handling. | Keep: profiling/monitoring and TensorBoard integration are wanted. |
| `signal/` | Removed | Signal/window helper surface. | Removed. |

## User-Facing PyTorch Compatibility

| Directory | Posture | What it is | Notes |
| --- | --- | --- | --- |
| `nn/` | Keep narrow | Minimal `Module`, `Parameter`, and disabled layer facade. | Many compatibility paths still import `nn`. Delete later only after serialization/JIT/package paths are settled. |
| `testing/` | Removed | PyTorch testing utilities. | Removed; `_refs` now owns its tiny `float64` precision constant and `library.opcheck` reports unavailable. |
| `package/` | Removed | PyTorch package importer/exporter support. | Removed; `_jit_internal` keeps a local mangled-name predicate so TorchScript compatibility helpers do not import package machinery. |
| `utils/` | Keep narrow | Large utility namespace: pytree, data, serialization helpers, benchmarking, tensorboard, visualization, etc. | Do not delete wholesale. Keep `tensorboard`, `viz`, `benchmark`, `_strobelight`, serialization/data/core helpers; `utils/_debug_mode`, `utils/jit`, `checkpoint`, `bundled_inputs`, `mobile_optimizer`, `mkldnn`, and `hipify` were removed as non-runtime compatibility leftovers. |
| `contrib/` | Keep narrow | Contributed helpers such as tensorboard visualization. | Keep because TensorBoard/profiling visualization is wanted. |
| `legacy/` | Removed | Legacy README placeholder. | Removed. |
| `bin/` | Keep narrow | Installed executable payload such as `torch_shm_manager`. | Related to multiprocessing/shared memory. Keep while multiprocessing is kept. |
| `futures/` | Keep narrow | Future API compatibility. | Listed above too because distributed depends on it. |

## Logging and Vendor

| Directory | Posture | What it is | Notes |
| --- | --- | --- | --- |
| `_logging/` | Keep narrow | Structured/logging registrations. | Some runtime logging imports distributed; can be narrowed. |
| `_strobelight/` | Keep narrow | Compile-time/function profiling diagnostic tooling. | Keep for now because profiling tooling is wanted; can be narrowed after profiler UX is mapped. |
| `_vendor/` | Keep | Vendored packaging helpers. | Small and low-risk; remove only if imports prove dead. |

## Build Payload

| Directory | Posture | What it is | Notes |
| --- | --- | --- | --- |
| `include/` | Build payload | Installed C++ headers: ATen, c10, torch/csrc, pybind11, FHE native headers. | Do not delete manually. Shrink by changing install/build rules. |
| `lib/` | Build payload | Installed libraries/cmake/pkgconfig payload. | Do not delete manually. |
| `lib64/` | Build payload | Installed lib64/cmake/pkgconfig payload. | Do not delete manually. |
| `share/` | Build payload | Installed CMake/share data. | Do not delete manually. |

## Already Removed In This Branch

| Directory | Status | Why |
| --- | --- | --- |
| `_awaits/` | Removed | JIT awaitable helpers after the JIT package was removed. |
| `_decomp/` | Removed | Decomposition tables were compiler/meta/functorch-adjacent and already broken through missing native functorch pieces. |
| `_dynamo/` | Removed | TorchDynamo graph-capture stack is outside EasyFHE runtime. |
| `_export/` | Removed | Internal export implementation after public `export/` was removed. |
| `_functorch/` | Removed | Functorch/vmap transform stack is outside runtime; public `func/` was already removed. |
| `_higher_order_ops/` | Removed | Higher-order operator compiler/functorch support removed with the transform stack. |
| `_inductor/` | Removed | Inductor compiler backend is outside runtime. |
| `_lazy/` | Removed | LazyTensor backend compatibility not needed. |
| `_native/` | Removed | PyTorch/Triton native prototype surface was unused and import-broken; not EasyFHE's `_C` native extension. |
| `_numpy/` | Removed | NumPy namespace emulation removed; native tensor/NumPy conversion remains. |
| `ao/` | Removed | Model optimization/quantization facade; disabled stubs. |
| `amp/` | Removed | Autocast/GradScaler compatibility surface; deleted with `cuda/amp/` and `cpu/amp/`. |
| `backends/_coreml/` | Removed | CoreML conversion support not needed. |
| `backends/cudnn/` | Removed | cuDNN compatibility surface not needed by current EasyFHE runtime. |
| `backends/cusparselt/` | Removed | cuSPARSELt backend query not needed by current FHE runtime. |
| `backends/kleidiai/` | Removed | ARM KleidiAI backend query not needed by current CUDA runtime. |
| `backends/mha/` | Removed | Multi-head attention fastpath flag not needed after model-training surfaces were removed. |
| `backends/miopen/` | Removed | ROCm/MIOpen compatibility surface not needed. |
| `backends/mkl/` | Removed | MKL backend query/verbose support not needed by current CUDA runtime. |
| `backends/mkldnn/` | Removed | MKLDNN compatibility surface not needed. |
| `backends/mps/` | Removed | MPS backend compatibility surface not needed. |
| `backends/nnpack/` | Removed | NNPACK compatibility surface not needed. |
| `backends/opt_einsum/` | Removed | Optional einsum contraction-path optimizer not needed. |
| `backends/python_native/` | Removed | Python-native DSL backend controls not needed. |
| `backends/quantized/` | Removed | Quantized backend compatibility not needed. |
| `backends/xeon/` | Removed | Xeon CPU launcher tooling not needed. |
| `backends/xnnpack/` | Removed | XNNPACK compatibility surface not needed. |
| `compiler/` | Removed | Public compiler/cache surface removed; top-level `easyfhe.compiler` disabled namespace remains for runtime guards. |
| `distributions/` | Removed | Probability distributions are outside encrypted inference runtime. |
| `export/` | Removed | Public export/PT2 archive workflows are not used. |
| `fft/` | Removed | Public FFT namespace outside FHE surface; internal refs remain. |
| `func/` | Removed | Public `torch.func` surface disabled. |
| `jit/` | Removed | TorchScript/JIT package removed. A tiny top-level `easyfhe.jit` stub remains for runtime checks. |
| `legacy/` | Removed | Legacy placeholder removed. |
| `linalg/` | Removed | Public namespace already failed because `_C._linalg` was absent. |
| `masked/` | Removed | MaskedTensor compatibility surface was not FHE-specific and import-broken. |
| `mps/` | Removed | Apple MPS backend surface not needed for Linux/CUDA-focused runtime. |
| `mtia/` | Removed | MTIA accelerator surface not needed. |
| `nested/` | Removed | NestedTensor compatibility surface was not FHE-specific and import-broken. |
| `numa/` | Removed | NUMA helper surface not needed by current runtime. |
| `onnx/` | Removed | ONNX export disabled and not used. |
| `optim/` | Removed | Training optimizers are outside inference runtime. |
| `package/` | Removed | PyTorch package importer/exporter support removed with JIT/export cleanup. |
| `quantization/` | Removed | Public quantization facade disabled. |
| `signal/` | Removed | Signal/window helpers not needed by current runtime. |
| `special/` | Removed | Public namespace already failed because `_C._special` was absent. |
| `testing/` | Removed | PyTorch testing utilities removed; tiny constants/fallbacks were moved to local runtime code. |
| `utils/_debug_mode/` | Removed | Debug mode stack was compiler/functorch-related and import-broken. |
| `utils/bundled_inputs.py` | Removed | TorchScript bundled-input helper removed with JIT/package cleanup. |
| `utils/checkpoint.py` | Removed | Training activation checkpointing was outside runtime and already import-broken. |
| `utils/hipify/` | Removed | ROCm hipify compatibility removed. |
| `utils/jit/` | Removed | JIT log/IR benchmarking helpers removed with JIT. |
| `utils/mkldnn.py` | Removed | MKLDNN conversion helper removed after MKLDNN backend compatibility was removed. |
| `utils/mobile_optimizer.py` | Removed | Mobile/JIT optimizer helper removed. |

## Trimmed To Anchors

| Directory | Status | Why |
| --- | --- | --- |
| `sparse/` | Keep minimal | Public sparse API implementation was removed, but `easyfhe.sparse` and `easyfhe.cuda.sparse` anchor modules remain because `_C._initExtension()` imports them while installing native sparse tensor classes. |

## Suggested Next Batches

1. Native/typing stub audit: inspect `_C_flatbuffer/` and stale `_C/*.pyi` references left by removed public surfaces.
2. Tensor-runtime narrowing only, no wholesale deletes: inspect `_subclasses/`, `_refs/`, `_prims/`, and `_library/` for now-broken compiler/functorch/sparse/nested hooks.
3. Utility narrowing: keep profiling/visualization (`profiler/`, `monitor/`, `utils/tensorboard/`, `utils/viz/`, `utils/benchmark/`, `_strobelight/`, `utils/_strobelight/`, `contrib/`), but inspect remaining non-core utility files one by one.
4. Distributed/NCCL follow-up: keep `distributed/`, `futures/`, `multiprocessing/`, `bin/`, and `cuda/nccl.py`; separately fix/re-enable NCCL build support.
5. Build payload trimming: adjust install rules for `include/`, `lib/`, `lib64/`, and `share/` after native dependency mapping is clear.

Minimum verification after each batch:

```bash
python3 -X faulthandler -c "import easyfhe; print(easyfhe.__version__)"
python3 -X faulthandler -c "import easyfhe.fhe as fhe; print(len(fhe.__all__))"
python3 -X faulthandler -c "import easyfhe.profiler; print('profiler ok')"
CUDA_VISIBLE_DEVICES=0 EASYFHE_DEVICE=cuda python3 -m examples.resnet20_aespa.main
```
