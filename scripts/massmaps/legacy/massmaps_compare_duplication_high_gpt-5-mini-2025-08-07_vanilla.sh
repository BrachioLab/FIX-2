#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" baselines/c2bd847_previous_pipeline/compare_massmaps_duplication_ablation.py \
  --variant duplicate_high \
  --method vanilla \
  --model gpt-5-mini-2025-08-07 \
  --eval-model gpt-5-mini-2025-08-07 \
  "$@"
