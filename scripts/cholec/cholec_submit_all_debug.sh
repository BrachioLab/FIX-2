#!/usr/bin/env bash
set -euo pipefail

scripts=(
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5.2-pro-2025-12-11_vanilla_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5.2-pro-2025-12-11_cot_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5.2-pro-2025-12-11_socratic_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5.2-pro-2025-12-11_subq_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5-mini-2025-08-07_vanilla_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5-mini-2025-08-07_cot_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5-mini-2025-08-07_socratic_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gpt-5-mini-2025-08-07_subq_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-opus-4-5-20251101_vanilla_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-opus-4-5-20251101_cot_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-opus-4-5-20251101_socratic_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-opus-4-5-20251101_subq_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-haiku-4-5-20251001_vanilla_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-haiku-4-5-20251001_cot_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-haiku-4-5-20251001_socratic_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_claude-haiku-4-5-20251001_subq_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-pro_vanilla_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-pro_cot_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-pro_socratic_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-pro_subq_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-flash_vanilla_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-flash_cot_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-flash_socratic_debug.sh"
  "/shared_data0/weiqiuy/FIX-2/scripts/cholec/cholec_run_gemini-2.5-flash_subq_debug.sh"
)

extra_args=(--overwrite_existing "$@")

mkdir -p logs

for script in "${scripts[@]}"; do
  log_file="logs/$(basename "${script}").log"
  echo "Starting ${script} (log: ${log_file})"
  nohup bash "${script}" "${extra_args[@]}" > "${log_file}" 2>&1 &
  echo "Started PID $!"
done
