#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" baselines/c2bd847_previous_pipeline/massmaps_duplication_ablation.py \
  --method vanilla \
  --model gpt-5-mini-2025-08-07 \
  --eval-model gpt-5-mini-2025-08-07 \
  --paraphrase-model gpt-5-mini-2025-08-07 \
  --num-samples 2 \
  "$@"
