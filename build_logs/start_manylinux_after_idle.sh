#!/usr/bin/env bash
set -euo pipefail

cd /home/zrji/EasyFHE

run_id="manylinux-0.1.1-$(date +%Y%m%d-%H%M%S)"
log="build_logs/${run_id}.log"

{
  echo "waiting_started=$(date -Is)"
  while ps -ef | grep -E 'runc .*buildkit|dnf .*cuda-toolkit' | grep -v grep >/dev/null; do
    echo "waiting_for_buildkit=$(date -Is)"
    ps -ef | grep -E 'runc .*buildkit|dnf .*cuda-toolkit' | grep -v grep || true
    sleep 60
  done
  echo "matrix_started=$(date -Is)"
  exec /home/zrji/EasyFHE/build_logs/run_manylinux_matrix.sh
} >"${log}" 2>&1
