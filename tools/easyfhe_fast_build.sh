#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export USE_EASYFHE_FAST_BUILD="${USE_EASYFHE_FAST_BUILD:-1}"
export USE_EASYFHE_FAST_INFERENCE="${USE_EASYFHE_FAST_INFERENCE:-1}"
export USE_CUDA="${USE_CUDA:-1}"
export USE_ROCM="${USE_ROCM:-0}"
export USE_NCCL="${USE_NCCL:-1}"
export USE_DISTRIBUTED="${USE_DISTRIBUTED:-1}"
export USE_GLOO="${USE_GLOO:-1}"
export USE_TENSORPIPE="${USE_TENSORPIPE:-1}"
export USE_MPI="${USE_MPI:-0}"
export USE_UCC="${USE_UCC:-0}"

export USE_CUDNN="${USE_CUDNN:-0}"
export USE_CUSPARSELT="${USE_CUSPARSELT:-0}"
export USE_CUDSS="${USE_CUDSS:-0}"
export USE_CUFILE="${USE_CUFILE:-0}"
export USE_NVSHMEM="${USE_NVSHMEM:-0}"
export USE_NVRTC="${USE_NVRTC:-0}"
export USE_MAGMA="${USE_MAGMA:-0}"

export USE_MKLDNN="${USE_MKLDNN:-0}"
export USE_FBGEMM="${USE_FBGEMM:-0}"
export USE_NNPACK="${USE_NNPACK:-0}"
export USE_QNNPACK="${USE_QNNPACK:-0}"
export USE_PYTORCH_QNNPACK="${USE_PYTORCH_QNNPACK:-0}"
export USE_XNNPACK="${USE_XNNPACK:-0}"

export USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-0}"
export USE_MEM_EFF_ATTENTION="${USE_MEM_EFF_ATTENTION:-0}"
export USE_KINETO="${USE_KINETO:-0}"
export USE_ITT="${USE_ITT:-0}"
export USE_OBSERVERS="${USE_OBSERVERS:-0}"
export USE_NUMA="${USE_NUMA:-0}"
export USE_VALGRIND="${USE_VALGRIND:-0}"

export BUILD_TEST="${BUILD_TEST:-0}"
export BUILD_CUSTOM_PROTOBUF="${BUILD_CUSTOM_PROTOBUF:-1}"
export BUILD_MOBILE_TEST="${BUILD_MOBILE_TEST:-0}"
export BUILD_MOBILE_BENCHMARK="${BUILD_MOBILE_BENCHMARK:-0}"
export BUILD_FUNCTORCH="${BUILD_FUNCTORCH:-0}"
export BUILD_LAZY_TS_BACKEND="${BUILD_LAZY_TS_BACKEND:-0}"
export BUILD_LAZY_CUDA_LINALG="${BUILD_LAZY_CUDA_LINALG:-0}"
export NO_API="${NO_API:-1}"

export USE_NINJA="${USE_NINJA:-1}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"
export CMAKE_FRESH="${CMAKE_FRESH:-1}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  first_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -n "${first_cap}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${first_cap}"
  fi
fi

if [[ "${USE_NINJA}" != "0" && -f build/CMakeCache.txt ]] && grep -q '^CMAKE_GENERATOR:INTERNAL=Unix Makefiles$' build/CMakeCache.txt; then
  if [[ "${EASYFHE_CLEAN_BUILD:-0}" == "1" ]]; then
    rm -rf build
  else
    echo "Existing build/ uses Unix Makefiles. Set EASYFHE_CLEAN_BUILD=1 to recreate it with Ninja." >&2
    echo "Continuing with the existing generator for this run." >&2
    export USE_NINJA=0
  fi
fi

echo "EasyFHE fast build:"
echo "  USE_EASYFHE_FAST_BUILD=${USE_EASYFHE_FAST_BUILD}"
echo "  USE_EASYFHE_FAST_INFERENCE=${USE_EASYFHE_FAST_INFERENCE}"
echo "  USE_CUDA=${USE_CUDA}"
echo "  USE_DISTRIBUTED=${USE_DISTRIBUTED}"
echo "  USE_NCCL=${USE_NCCL}"
echo "  USE_GLOO=${USE_GLOO}"
echo "  USE_TENSORPIPE=${USE_TENSORPIPE}"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo "  CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
echo "  USE_NINJA=${USE_NINJA}"
echo "  MAX_JOBS=${MAX_JOBS}"

python3 setup.py develop "$@"
