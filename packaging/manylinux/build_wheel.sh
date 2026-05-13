#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_VERSION="${CUDA_VERSION:-12.9}"
CUDA_PACKAGE_SUFFIX="${CUDA_PACKAGE_SUFFIX:-${CUDA_VERSION/./-}}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cu${CUDA_VERSION/./}}"
MANYLINUX_PLAT="${MANYLINUX_PLAT:-manylinux_2_28_x86_64}"
MANYLINUX_IMAGE="${MANYLINUX_IMAGE:-quay.io/pypa/${MANYLINUX_PLAT}}"
IMAGE="${IMAGE:-easyfhe-${MANYLINUX_PLAT}-${CUDA_FLAVOR}}"
DOCKERFILE="${DOCKERFILE:-${ROOT_DIR}/packaging/manylinux/Dockerfile.cuda12.9}"
PYTHON_TAG="${PYTHON_TAG:-cp312-cp312}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"
WHEELHOUSE="${WHEELHOUSE:-${ROOT_DIR}/wheelhouse/${MANYLINUX_PLAT}/${CUDA_FLAVOR}}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-}"
DOCKER_RUN_NETWORK="${DOCKER_RUN_NETWORK:-}"

if ! command -v "${CONTAINER_ENGINE}" >/dev/null 2>&1; then
  echo "Container engine '${CONTAINER_ENGINE}' was not found. Set CONTAINER_ENGINE=docker or CONTAINER_ENGINE=podman." >&2
  exit 2
fi

build_network_args=()
run_network_args=()
cuda_base_image_args=()
if [[ -n "${DOCKER_BUILD_NETWORK}" ]]; then
  build_network_args+=(--network "${DOCKER_BUILD_NETWORK}")
fi
if [[ -n "${DOCKER_RUN_NETWORK}" ]]; then
  run_network_args+=(--network "${DOCKER_RUN_NETWORK}")
fi
if [[ -n "${CUDA_BASE_IMAGE:-}" ]]; then
  cuda_base_image_args+=(--build-arg CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE}")
fi

"${CONTAINER_ENGINE}" build \
  -f "${DOCKERFILE}" \
  "${build_network_args[@]}" \
  "${cuda_base_image_args[@]}" \
  --build-arg MANYLINUX_IMAGE="${MANYLINUX_IMAGE}" \
  --build-arg MANYLINUX_PLAT="${MANYLINUX_PLAT}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg CUDA_PACKAGE_SUFFIX="${CUDA_PACKAGE_SUFFIX}" \
  --build-arg CUDA_DNF_PACKAGES="${CUDA_DNF_PACKAGES:-cuda-toolkit-${CUDA_PACKAGE_SUFFIX}}" \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
  --build-arg NO_PROXY="${NO_PROXY:-${no_proxy:-}}" \
  --build-arg http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
  --build-arg https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
  --build-arg no_proxy="${no_proxy:-${NO_PROXY:-}}" \
  -t "${IMAGE}" \
  "${ROOT_DIR}/packaging/manylinux"

mkdir -p "${WHEELHOUSE}"

"${CONTAINER_ENGINE}" run --rm \
  "${run_network_args[@]}" \
  -e PYTHON_TAG="${PYTHON_TAG}" \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  -e MAX_JOBS="${MAX_JOBS}" \
  -e CUDA_FLAVOR="${CUDA_FLAVOR}" \
  -e CUDA_VERSION="${CUDA_VERSION}" \
  -e AUDITWHEEL_PLAT="${MANYLINUX_PLAT}" \
  -e USE_AVX="${USE_AVX:-0}" \
  -e CLEAN_BUILD="${CLEAN_BUILD:-1}" \
  -e AUDITWHEEL_EXCLUDE_LIBS="${AUDITWHEEL_EXCLUDE_LIBS:-}" \
  -e HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
  -e HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
  -e NO_PROXY="${NO_PROXY:-${no_proxy:-}}" \
  -e http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
  -e https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
  -e no_proxy="${no_proxy:-${NO_PROXY:-}}" \
  -v "${ROOT_DIR}:/io" \
  -v "${WHEELHOUSE}:/wheelhouse" \
  "${IMAGE}"
