#!/usr/bin/env bash
set -euo pipefail

PYTHON_TAG="${PYTHON_TAG:-cp312-cp312}"
PYTHON_BIN="/opt/python/${PYTHON_TAG}/bin/python"
PLAT="${AUDITWHEEL_PLAT:-manylinux_2_28_x86_64}"
WHEELHOUSE="${WHEELHOUSE:-/wheelhouse}"
DIST_DIR="${DIST_DIR:-/io/dist}"
CUDA_VERSION="${CUDA_VERSION:-12.9}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cu${CUDA_VERSION/./}}"

source /io/packaging/manylinux/cuda_arch_list.sh

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python '${PYTHON_TAG}' was not found under /opt/python." >&2
  echo "Available interpreters:" >&2
  ls -1 /opt/python >&2
  exit 2
fi

mkdir -p "${WHEELHOUSE}" "${DIST_DIR}"

if command -v dnf >/dev/null 2>&1; then
  dnf_proxy="${http_proxy:-${HTTP_PROXY:-}}"
  if [[ -n "${dnf_proxy}" ]] && ! grep -q '^proxy=' /etc/dnf/dnf.conf; then
    echo "proxy=${dnf_proxy}" >> /etc/dnf/dnf.conf
  fi
  dnf install -y protobuf-compiler protobuf-devel
fi

"${PYTHON_BIN}" -m pip install --upgrade pip "setuptools<82" wheel auditwheel numpy packaging pyyaml requests six typing-extensions

git config --global --add safe.directory /io >/dev/null 2>&1 || true

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(easyfhe_default_cuda_arch_list "${CUDA_VERSION}")}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export USE_CUDA="${USE_CUDA:-1}"
export USE_CUDNN="${USE_CUDNN:-0}"
export USE_CUSPARSELT="${USE_CUSPARSELT:-0}"
export USE_CUDSS="${USE_CUDSS:-0}"
export USE_CUFILE="${USE_CUFILE:-0}"
export BUILD_TEST="${BUILD_TEST:-0}"
export USE_AVX="${USE_AVX:-0}"
export USE_EASYFHE_FAST_BUILD="${USE_EASYFHE_FAST_BUILD:-1}"
export USE_EASYFHE_FAST_INFERENCE="${USE_EASYFHE_FAST_INFERENCE:-1}"
export BUILD_CUSTOM_PROTOBUF="${BUILD_CUSTOM_PROTOBUF:-0}"
export USE_DISTRIBUTED="${USE_DISTRIBUTED:-0}"
export USE_NCCL="${USE_NCCL:-0}"

if [[ -z "${PYTORCH_BUILD_VERSION:-}" ]]; then
  base_version="$(tr -d '[:space:]' < /io/version.txt)"
  if git -C /io rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_sha="$(git -C /io rev-parse --short=7 HEAD)"
    export PYTORCH_BUILD_VERSION="${base_version}+${CUDA_FLAVOR}.git${git_sha}"
  else
    export PYTORCH_BUILD_VERSION="${base_version}+${CUDA_FLAVOR}"
  fi
  export PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER:-1}"
fi

if [[ "${PYTHON_ONLY_REPACK:-0}" == "1" ]]; then
  "${PYTHON_BIN}" /io/packaging/manylinux/repack_python_wheel.py \
    --repo /io \
    --wheelhouse "${WHEELHOUSE}" \
    --python-tag "${PYTHON_TAG}" \
    --cuda-flavor "${CUDA_FLAVOR}" \
    --plat "${PLAT}" \
    --version "${PYTORCH_BUILD_VERSION}"
  ls -lh "${WHEELHOUSE}"
  exit 0
fi

if [[ "${CLEAN_BUILD:-1}" == "1" ]]; then
  rm -rf /io/build
fi
rm -rf /io/dist/*
"${PYTHON_BIN}" setup.py bdist_wheel

wheel="$(ls -1 /io/dist/easyfhe-*.whl | tail -n 1)"
echo "Built wheel: ${wheel}"

auditwheel show "${wheel}" || true

repair_args=(
  repair
  --plat "${PLAT}"
  --wheel-dir "${WHEELHOUSE}"
)

# CUDA wheels usually should not vendor the kernel driver library. The CUDA
# runtime policy is still a release decision: either exclude runtime libraries
# and document the CUDA requirement, or depend on NVIDIA's PyPI CUDA packages.
for lib in ${AUDITWHEEL_EXCLUDE_LIBS:-libcuda.so.1 libcuda.so libcudart.so.12 libcudart.so.13 libcublas.so.12 libcublas.so.13 libcufft.so.11 libcufft.so.12 libcurand.so.10 libcurand.so.11 libcusparse.so.12 libcusparse.so.13 libcusolver.so.11 libcusolver.so.12 libnvrtc.so.12 libnvrtc.so.13}; do
  repair_args+=(--exclude "${lib}")
done

auditwheel "${repair_args[@]}" "${wheel}" || {
  echo "auditwheel repair failed; keeping the original linux wheel in ${DIST_DIR}." >&2
  cp -f "${wheel}" "${WHEELHOUSE}/"
}

ls -lh "${WHEELHOUSE}"
