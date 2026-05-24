#!/usr/bin/env bash
set -euo pipefail

cd /home/zrji/EasyFHE

echo "started=$(date -Is)"
echo "host=$(hostname)"
echo "nproc=$(nproc)"

export MAX_JOBS=24
export MANYLINUX_PLAT=manylinux_2_28_x86_64
export CLEAN_BUILD=1
export USE_AVX=0
export USE_MSLK=0
export DOCKER_BUILD_NETWORK=host
export DOCKER_RUN_NETWORK=host

for cuda_version in 12.9 13.0 13.2; do
  cuda_flavor="cu${cuda_version/./}"
  if [[ "${cuda_version}" == "12.4" ]]; then
    arch_list="7.5 8.0 8.6 8.9 9.0"
  else
    arch_list="7.5 8.0 8.6 8.9 9.0 10.0 12.0"
  fi

  for py_tag in cp310-cp310 cp312-cp312; do
    echo "=== START ${cuda_flavor} ${py_tag} arch=${arch_list} $(date -Is) ==="
    env \
      CUDA_VERSION="${cuda_version}" \
      CUDA_PACKAGE_SUFFIX="${cuda_version/./-}" \
      CUDA_FLAVOR="${cuda_flavor}" \
      PYTORCH_BUILD_VERSION="0.1.1+${cuda_flavor}" \
      PYTORCH_BUILD_NUMBER="1" \
      PYTHON_TAG="${py_tag}" \
      TORCH_CUDA_ARCH_LIST="${arch_list}" \
      WHEELHOUSE="/home/zrji/EasyFHE/wheelhouse/manylinux_2_28_x86_64/${cuda_flavor}" \
      /home/zrji/EasyFHE/packaging/manylinux/build_wheel.sh
    echo "=== DONE ${cuda_flavor} ${py_tag} $(date -Is) ==="
  done
done

echo "finished=$(date -Is)"
find /home/zrji/EasyFHE/wheelhouse/manylinux_2_28_x86_64 \
  -maxdepth 2 \
  -type f \
  -name "easyfhe-*.whl" \
  -ls
