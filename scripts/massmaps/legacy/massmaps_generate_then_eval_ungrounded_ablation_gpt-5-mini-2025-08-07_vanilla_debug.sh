#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ABLATION_SCRIPT="baselines/c2bd847_previous_pipeline/massmaps_ungrounded_ablation.py"

"${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
  --method vanilla \
  --model gpt-5-mini-2025-08-07 \
  --eval-model gpt-5-mini-2025-08-07 \
  --generation-model gpt-5-mini-2025-08-07 \
  --num-samples 2 \
  --generate-only \
  "$@"

"${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
  --method vanilla \
  --model gpt-5-mini-2025-08-07 \
  --eval-model gpt-5-mini-2025-08-07 \
  --generation-model gpt-5-mini-2025-08-07 \
  --num-samples 2 \
  --evaluate-only \
  "$@"
