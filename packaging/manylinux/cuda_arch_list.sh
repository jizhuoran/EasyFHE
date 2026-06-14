#!/usr/bin/env bash

easyfhe_default_cuda_arch_list() {
  local cuda_version="${1:-}"
  local major="${cuda_version%%.*}"
  local rest="${cuda_version#*.}"
  local minor="${rest%%.*}"

  if [[ ! "${major}" =~ ^[0-9]+$ || ! "${minor}" =~ ^[0-9]+$ ]]; then
    echo "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
    return
  fi

  if (( major > 12 || (major == 12 && minor >= 8) )); then
    echo "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
  else
    echo "7.5;8.0;8.6;8.9;9.0+PTX"
  fi
}
