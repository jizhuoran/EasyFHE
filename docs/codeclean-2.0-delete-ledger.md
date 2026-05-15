# codeclean-2.0 delete ledger

This file tracks Python surface removals and the native C++/CUDA follow-up
anchors they imply. The goal is to avoid deleting a Python namespace and then
forgetting the generated bindings, headers, kernels, or CMake switches that
exist only to support that namespace.

## Current removal

| Status | Python path | Why | Native follow-up anchors |
| --- | --- | --- | --- |
| Deleted | `easyfhe/fft/__init__.py` | Public `easyfhe.fft` is outside the FHE runtime surface and the AESPA ResNet20 example passes without it. | `tools/autograd/gen_python_functions.py` (`is_py_fft_function`, `torch.fft`, `THPFFTVariableFunctionsModule`), `tools/autograd/templates/python_fft_functions.cpp`, `torch/csrc/autograd/python_fft_functions.h`, `torch/csrc/autograd/generated/python_fft_functions.cpp`, `torch/csrc/Module.cpp`, `tools/autograd/gen_annotated_fn_args.py`, `aten/src/ATen/native/SpectralOpsUtils.h`, `aten/src/ATen/core/interned_strings.h` (`_fft_c2c`). |
| Removed lazy import | `easyfhe.__init__` `fft` entry | Prevents `from easyfhe import fft` from advertising a deleted public module. | Same as above. |
| Deleted | `easyfhe/ao/**`, `easyfhe/quantization/__init__.py` | These were disabled stubs and are outside the FHE runtime surface. | `aten/src/ATen/native/easyfhe_quantized_stubs.*`, `aten/src/ATen/quantized/**`, `aten/src/ATen/native/*quant*`, generated op headers containing `quantized`, `CMakeLists.txt` quantization-related options if any. |
| Deleted | `easyfhe/distributions/__init__.py` | Disabled stub; probability distributions are not part of encrypted inference runtime. | `aten/src/ATen/native/*Distribution*`, CPU/CUDA random/distribution kernels if they exist only for public distributions, generated docs/types that mention `torch.distributions`. Keep tensor factory RNG until proven unused. |
| Deleted | `easyfhe/func/__init__.py` | Disabled `torch.func`-style surface. | `aten/src/ATen/functorch/easyfhe_functorch_stubs.cpp`, `easyfhe/_functorch/**`, `easyfhe/_higher_order_ops/**`, `torch/csrc/dynamo/**`, generated functorch/vmap glue. |
| Deleted | `easyfhe/onnx/__init__.py` | Export was disabled; missing ONNX submodule already complicates full configure. | `torch/csrc/onnx/**`, `third_party/onnx`, `cmake/Dependencies.cmake` ONNX block, `CMakeLists.txt` `USE_SYSTEM_ONNX`, `cmake/PreBuildSteps.cmake`. |
| Deleted | `easyfhe/export/**` | PyTorch export/PT2 archive workflows are not part of the EasyFHE runtime surface. | `easyfhe/_decomp/__init__.py` imports `easyfhe.export.decomp_utils` and `easyfhe.export.exported_program`, `easyfhe/_library/triton.py` lazily imports `easyfhe.export._trace`, `easyfhe/_export/**`, `torch/csrc/jit/serialization/**`, `_dynamo/_inductor` export-related paths. |
| Deleted | `easyfhe/jit/**` | TorchScript/JIT workflows are outside the FHE inference surface. `easyfhe.__init__` keeps a small top-level `easyfhe.jit` stub for runtime checks such as `torch.jit.is_scripting()`. | `torch/csrc/jit/**`, `torch/csrc/Module.cpp`, `easyfhe/_jit_internal.py`, `easyfhe/utils/bundled_inputs.py`, serialization/pickler paths. |
| Deleted | `easyfhe/optim/**` | Training optimizers are outside the FHE inference surface. `easyfhe.profiler` was decoupled from the optimizer-step hook so profiler can remain. | `easyfhe/profiler/__init__.py`, optimizer hook registration, training examples under `examples/lstm/**`, generated optimizer docs/types if any. |
| Deleted | `easyfhe/amp/**`, `easyfhe/cuda/amp/**`, `easyfhe/cpu/amp/**` | Autocast/GradScaler compatibility is outside the FHE inference surface. | `easyfhe.__init__` no longer imports `amp`, `autocast`, or `GradScaler`; `easyfhe/cuda/__init__.py` and `easyfhe/cpu/__init__.py` no longer import their `amp` subpackages. |
| Deleted | `easyfhe/mps/**`, `easyfhe/mtia/**`, `easyfhe/numa/**`, `easyfhe/signal/**` | Non-CUDA backend surfaces and signal/window helpers are outside the current Linux/CUDA-focused runtime. | `easyfhe/random.py` no longer imports `mps`/`mtia` during seeding; top-level lazy imports for `mps`/`mtia` were removed. |
| Deleted | `easyfhe/backends/cudnn/**`, `easyfhe/backends/mkldnn/**`, `easyfhe/backends/nnpack/**`, `easyfhe/backends/mps/**`, `easyfhe/backends/miopen/**`, `easyfhe/backends/quantized/**`, `easyfhe/backends/_coreml/**`, `easyfhe/backends/xeon/**`, `easyfhe/backends/xnnpack/**`, `easyfhe/backends/python_native/**` | Unsupported or unused backend compatibility surfaces. | `easyfhe/backends/__init__.py` no longer imports them; `easyfhe.__init__` no longer calls `torch.backends.mps._init()` or exposes deprecated `has_mps/has_cudnn/has_mkldnn`; `_utils`, `cpp_extension`, and `collect_env` no longer hard-reference MPS/XNNPACK backend modules. |
| Deleted | `easyfhe/backends/cusparselt/**`, `easyfhe/backends/kleidiai/**`, `easyfhe/backends/mha/**`, `easyfhe/backends/mkl/**`, `easyfhe/backends/opt_einsum/**` | Remaining optional backend compatibility surfaces not needed by the current FHE runtime. | `_meta_registrations.py` treats MKL/KleidiAI backend checks as unavailable; `functional.py` and `_numpy/_funcs_impl.py` no longer use the opt-einsum backend and fall back to default `einsum` behavior. |
| Deleted | `easyfhe/_dynamo/**`, `easyfhe/_inductor/**`, `easyfhe/_export/**`, `easyfhe/compiler/**`, `easyfhe/_lazy/**`, `easyfhe/_awaits/**` | First compiler-stack cleanup batch. These are graph capture, export, compile backend, lazy backend, and JIT awaitable surfaces outside the current tensor/CUDA/FHE runtime. | `easyfhe.__init__` removed lazy imports for `_dynamo/_inductor/_export`, keeps a small in-memory `easyfhe.compiler` namespace for `is_compiling()` and CUDA graph config checks, and keeps `compile()` disabled. `easyfhe._jit_internal` keeps a local `_Await` placeholder. Native follow-up anchors: `torch/csrc/dynamo/**`, `torch/csrc/jit/python/pybind_utils.h` await import, generated `_C/_dynamo/*.pyi`, Inductor/AOTInductor CMake or packaging references, export serde/PT2 archive references. |
| Deleted | `easyfhe/linalg/__init__.py`, `easyfhe/special/__init__.py` | These public namespaces already failed on import because `_C._linalg` and `_C._special` are not present. | `tools/autograd/gen_python_functions.py`, `tools/autograd/templates/python_linalg_functions.cpp`, `tools/autograd/templates/python_special_functions.cpp`, `torch/csrc/autograd/python_{linalg,special}_functions.h`, `torch/csrc/autograd/generated/python_{linalg,special}_functions.cpp`, `torch/csrc/Module.cpp`, `aten/src/ATen/native/cuda/linalg/**`, `aten/src/ATen/native/*special*`, `aten/src/ATen/native/cuda/*bessel*`, `aten/src/ATen/native/cuda/*chebyshev*`. |
| Removed lazy imports | `easyfhe.__init__` entries for `onnx`, `distributions`, `func`, `linalg`, `special` | Prevents top-level `easyfhe` attribute lookup from advertising deleted modules. | Same anchors as the deleted modules above. |

Notes:

- `easyfhe._refs.fft` and `easyfhe._numpy.fft` still exist. They are internal
  PyTorch compatibility layers, not the public `easyfhe.fft` package.
- External compatibility risk: code using `import easyfhe.fft` will now fail.

## Import probe

After the Python 3.12 native rebuild:

| Module | Current result | Cleanup read |
| --- | --- | --- |
| `easyfhe.fft` | `ModuleNotFoundError` | Expected after deletion. |
| `easyfhe.distributions` | `ModuleNotFoundError` | Deleted. |
| `easyfhe.quantization` | `ModuleNotFoundError` | Deleted. |
| `easyfhe.ao` | `ModuleNotFoundError` | Deleted. |
| `easyfhe.func` | `ModuleNotFoundError` | Deleted. |
| `easyfhe.onnx` | `ModuleNotFoundError` | Deleted. |
| `easyfhe.optim` | `ModuleNotFoundError` | Deleted. |
| `easyfhe.export` | `ModuleNotFoundError` | Deleted. |
| `easyfhe.nn` | imports | Training layers disabled, but `Module` and `Parameter` are used by serialization/JIT/examples. Medium to high risk. |
| `easyfhe.linalg` | `ModuleNotFoundError` | Deleted after it was found broken. |
| `easyfhe.sparse` | imports | C-backed sparse surface with fallbacks. Medium risk. |
| `easyfhe.special` | `ModuleNotFoundError` | Deleted after it was found broken. |
| `easyfhe.compiler` | `ModuleNotFoundError` | Deleted as an importable package; top-level `easyfhe.compiler` remains a disabled namespace for runtime guards. |
| `easyfhe._dynamo` | `ModuleNotFoundError` | Deleted. |
| `easyfhe._inductor` | `ModuleNotFoundError` | Deleted. |
| `easyfhe._export` | `ModuleNotFoundError` | Deleted. |
| `easyfhe._lazy` | `ModuleNotFoundError` | Deleted. |
| `easyfhe._awaits` | `ModuleNotFoundError` | Deleted. |

## Deleted Python batch verification

The following batch was deleted together and still passed the GPU AESPA ResNet20
smoke test:

```text
easyfhe.ao
easyfhe.quantization
easyfhe.distributions
easyfhe.func
easyfhe.onnx
easyfhe.export
easyfhe.jit
easyfhe.optim
easyfhe.linalg
easyfhe.special
```

Verification result:

```text
CUDA_VISIBLE_DEVICES=0 EASYFHE_DEVICE=cuda python3 -m examples.resnet20_aespa.main
accuracy: 1/1 (100.00%)
device: cuda
```

When `easyfhe/jit/**` was first removed, the GPU smoke test failed in
`torch.cuda.synchronize()` because `easyfhe/cuda/_utils.py` calls
`torch.jit.is_scripting()`. The fix is intentionally narrow: keep no
`easyfhe.jit` importable package, but keep a top-level `easyfhe.jit` stub object
with `is_scripting()`, `is_tracing()`, `annotate()`, and decorator no-ops. Real
TorchScript APIs such as `script`, `trace`, `freeze`, and `load` raise a runtime
error.

## Medium-risk candidates

| Candidate | Python paths | Why not first |
| --- | --- | --- |
| FX/functorch transform internals | `easyfhe/fx/**`, `easyfhe/_functorch/**`, `easyfhe/_higher_order_ops/**` | They sit above tensor dispatch but below public compiler APIs. Delete as the next separate batch because distributed functional collectives and fake tensor helpers still reference them. |
| Decomposition/custom-op edge | `easyfhe/_decomp/**`, `easyfhe/_custom_op/**` | These are compiler-adjacent but closer to operator/library infrastructure. Inspect separately so tensor/op registration is not accidentally damaged. |
| NN training surface | `easyfhe/nn/**` | Layers are disabled, but `Module`, `Parameter`, serialization, JIT, and weight-generation examples still import `easyfhe.nn`. This should be stubbed narrower before deletion. |
| Sparse surface | `easyfhe/sparse/__init__.py`, sparse internals | It imports today and some tensor/runtime code still knows about sparse layout. Native cleanup touches generated sparse bindings and ATen sparse kernels. |
| Distributed/futures/multiprocessing | `easyfhe/distributed/**`, `easyfhe/futures/**`, `easyfhe/multiprocessing/**` | Fast build can disable distributed, but Python pieces still exist and some profiler/pickler guards reference distributed macros. Remove after locking build flags to `USE_DISTRIBUTED=0`, `USE_NCCL=0`. |
| Package/testing | `easyfhe/package/**`, `easyfhe/testing/**` | Large PyTorch compatibility area. Native stubs already exist in places, but imports are cross-linked with serialization and `_C` initialization. |

## Keep for now

| Path | Reason |
| --- | --- |
| `easyfhe/fhe/**` | Current project surface. |
| `easyfhe/profiler/**` | Keep: project needs profiler support for runtime/performance investigation. The optimizer-step hook was removed when `easyfhe/optim/**` was deleted. |
| `easyfhe/distributed/**`, `easyfhe/cuda/nccl.py`, `torch/csrc/cuda/python_nccl.cpp`, `torch/csrc/cuda/comm.cpp`, `torch/csrc/distributed/c10d/**` | Keep: NCCL/broadcast/all-reduce primitives are needed. Current build has `USE_NCCL=OFF`, so enabling this path is a separate build task. |
| `easyfhe/cuda/**`, `easyfhe/cpu/**`, `easyfhe/autograd/**`, `easyfhe/_C*`, `easyfhe/_prims*`, `easyfhe/_refs/**` | Tensor runtime support; prune only with focused import/runtime tests. |
| `easyfhe/include/**`, `easyfhe/lib/**`, `easyfhe/lib64/**`, `easyfhe/share/**` | Installed header/library/share payload. Shrink through build/install rules first, not by ad hoc Python package deletion. |

## Native search commands

Useful follow-up commands before deleting native pieces:

```bash
rg -n "python_fft_functions|THPFFT|_C\\._fft|_fft" tools torch aten cmake CMakeLists.txt
rg -n "python_linalg_functions|THPLinalg|_C\\._linalg|linalg_" tools torch aten cmake CMakeLists.txt
rg -n "python_special_functions|THPSpecial|_C\\._special|special_" tools torch aten cmake CMakeLists.txt
rg -n "python_sparse_functions|THPSparse|_C\\._sparse|sparse" tools torch aten cmake CMakeLists.txt
rg -n "onnx|USE_SYSTEM_ONNX|third_party/onnx" tools torch aten cmake CMakeLists.txt
rg -n "USE_DISTRIBUTED|USE_NCCL|c10d|distributed" tools torch aten cmake CMakeLists.txt
rg -n "quantization|quantized|QTensor|Quantizer" tools torch aten cmake CMakeLists.txt
```

Minimum verification loop for each Python deletion batch:

```bash
python3 -X faulthandler -c "import easyfhe; print(easyfhe.__version__)"
python3 -X faulthandler -c "import easyfhe.fhe as fhe; print(len(fhe.__all__))"
CUDA_VISIBLE_DEVICES=0 EASYFHE_DEVICE=cuda python3 -m examples.resnet20_aespa.main
```
