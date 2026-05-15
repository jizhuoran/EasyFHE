# codeclean-2.0 native import note

## Symptom

After removing the empty `easyfhe.fft` package and importing `easyfhe`, Python crashed at:

```text
from easyfhe._C import *
```

`python3 -X faulthandler -c "import easyfhe"` reported a fatal segmentation fault. A gdb backtrace pointed into `libtorch_python.so`, around pybind11 signature generation while registering `torch::autograd::initEnumTag`.

## Root cause

The crash was not caused by `easyfhe.fft`. It exposed a stale native build:

```text
Python_EXECUTABLE=/opt/python/cp310-cp310/bin/python
_Python_INCLUDE_DIR=/opt/python/cp310-cp310/include/python3.10
```

The active runtime is Python 3.12, but the CMake cache and older build output were for Python 3.10. Loading the Python 3.10-built `libtorch_python` extension from Python 3.12 caused the ABI crash.

## Fix

Reconfigure and rebuild the native extension for the active Python runtime:

```bash
rm -f build/.ninja_log build/.ninja_deps
CMAKE_ONLY=1 BUILD_CUSTOM_PROTOBUF=OFF MAX_JOBS=4 tools/easyfhe_fast_build.sh
CMAKE_FRESH=0 BUILD_CUSTOM_PROTOBUF=OFF MAX_JOBS=4 tools/easyfhe_fast_build.sh
```

The fast build profile avoids optional third-party submodules that are not present in this checkout and now enables NCCL by default. `cmake/FileMirroring.cmake` also excludes `__pycache__` directories during install so stale root-owned `.pyc` files do not block the editable wheel install.

## Verification

These commands pass after the rebuild:

```bash
python3 -X faulthandler -c "import easyfhe; print('imported', easyfhe.__version__)"
python3 -X faulthandler -c "import easyfhe.fhe as fhe; print('fhe imported', len(fhe.__all__), hasattr(fhe, 'generate_context'))"
CUDA_VISIBLE_DEVICES=0 EASYFHE_DEVICE=cuda python3 -m examples.resnet20_aespa.main
```

The ResNet20 AESPA example completed on GPU with `device: cuda`, processed one CIFAR-10 image, and reported `accuracy: 1/1 (100.00%)`.

The `pynvml` deprecation warning seen during import is non-fatal and unrelated to this crash.

## FFT cleanup finding

Removing `easyfhe/fft/__init__.py` removes the public `easyfhe.fft` namespace. The main risk is external compatibility: code that imports `easyfhe.fft` or `from easyfhe import fft` will now fail.

This does not remove every FFT-related internal file. The project still has PyTorch compatibility internals such as `easyfhe/_refs/fft.py` and `easyfhe/_numpy/fft.py`, and other compatibility code may reference `torch.fft`. Those paths are separate from the public `easyfhe.fft` package and were not needed by the AESPA ResNet20 example.

For the current FHE-focused cleanup, deleting the public `easyfhe.fft` namespace is validated by:

```bash
CUDA_VISIBLE_DEVICES=0 EASYFHE_DEVICE=cuda python3 -m examples.resnet20_aespa.main
```

If broader PyTorch compatibility is still a goal, keep or stub `easyfhe.fft`. If the project is intentionally narrowing to the FHE runtime, deleting it is reasonable and the remaining internal FFT files can be evaluated in a later cleanup pass.
