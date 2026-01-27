#!/usr/bin/env bash
set -euo pipefail

scripts=(
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-haiku-4-5-20251001_cot.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-haiku-4-5-20251001_socratic.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-haiku-4-5-20251001_subq.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-haiku-4-5-20251001_vanilla.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-opus-4-5-20251101_cot.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-opus-4-5-20251101_socratic.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-opus-4-5-20251101_subq.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_claude-opus-4-5-20251101_vanilla.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-flash_cot.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-flash_socratic.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-flash_subq.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-flash_vanilla.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-pro_cot.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-pro_socratic.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-pro_subq.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gemini-2.5-pro_vanilla.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5-mini-2025-08-07_cot.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5-mini-2025-08-07_socratic.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5-mini-2025-08-07_subq.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5-mini-2025-08-07_vanilla.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5.2-pro-2025-12-11_cot.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5.2-pro-2025-12-11_socratic.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5.2-pro-2025-12-11_subq.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/massmaps/massmaps_run_gpt-5.2-pro-2025-12-11_vanilla.sh"
)

extra_args=(--overwrite_existing "$@")

mkdir -p logs

for script in "${scripts[@]}"; do
  log_file="logs/$(basename "${script}").log"
  echo "Starting ${script} (log: ${log_file})"
  nohup bash "${script}" "${extra_args[@]}" > "${log_file}" 2>&1 &
  echo "Started PID $!"
done
