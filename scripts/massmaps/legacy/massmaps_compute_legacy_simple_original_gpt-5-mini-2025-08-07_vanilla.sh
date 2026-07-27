#!/usr/bin/env bash
set -euo pipefail

EXPLANATION_MODEL="gpt-5-mini-2025-08-07"
EVALUATION_MODEL="gpt-5-mini-2025-08-07"
METHOD="vanilla"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" baselines/c2bd847_previous_pipeline/compute_massmaps_result_summary.py \
  --method "${METHOD}" \
  --model "${EXPLANATION_MODEL}" \
  --eval-model "${EVALUATION_MODEL}" \
  "$@"
