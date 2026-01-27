#!/usr/bin/env bash
set -euo pipefail

scripts=(
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_eval_gpt-5-mini-2025-08-07_cot.sh"
)

extra_args=("$@")

mkdir -p logs

for script in "${scripts[@]}"; do
  log_file="logs/$(basename "${script}").log"
  echo "Starting ${script} (log: ${log_file})"
  nohup bash "${script}" "${extra_args[@]}" > "${log_file}" 2>&1 &
  echo "Started PID $!"
done
