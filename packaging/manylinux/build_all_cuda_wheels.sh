#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_VERSIONS="${CUDA_VERSIONS:-12.4 12.6 12.8 12.9 13.2}"
MANYLINUX_PLAT="${MANYLINUX_PLAT:-manylinux_2_28_x86_64}"

for cuda_version in ${CUDA_VERSIONS}; do
  cuda_flavor="cu${cuda_version/./}"
  echo "=== Building EasyFHE ${MANYLINUX_PLAT} ${cuda_flavor} wheel ==="
  CUDA_VERSION="${cuda_version}" \
  CUDA_PACKAGE_SUFFIX="${cuda_version/./-}" \
  CUDA_FLAVOR="${cuda_flavor}" \
  MANYLINUX_PLAT="${MANYLINUX_PLAT}" \
  WHEELHOUSE="${ROOT_DIR}/wheelhouse/${MANYLINUX_PLAT}/${cuda_flavor}" \
    "${ROOT_DIR}/packaging/manylinux/build_wheel.sh"
done

find "${ROOT_DIR}/wheelhouse" -maxdepth 3 -type f -name 'easyfhe-*.whl' -print
