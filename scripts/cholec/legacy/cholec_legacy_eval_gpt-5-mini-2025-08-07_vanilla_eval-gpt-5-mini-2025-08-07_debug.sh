#!/usr/bin/env bash
set -euo pipefail

EXPLANATION_MODEL="gpt-5-mini-2025-08-07"
EVALUATION_MODEL="gpt-5-mini-2025-08-07"
METHOD="vanilla"
NUM_SAMPLES="5"
LEGACY_NAME="legacy-c2bd847"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_PATH="results/${METHOD}/cholec_${EXPLANATION_MODEL}.json"
OUTPUT_DIR="baselines/c2bd847_previous_pipeline/notebooks/_dump/cholec/final/${LEGACY_NAME}/${EXPLANATION_MODEL}/${METHOD}/eval.${EVALUATION_MODEL}"

"${PYTHON_BIN}" baselines/c2bd847_previous_pipeline/run_baseline.py \
  --dataset cholec \
  --method "${METHOD}" \
  --model "${EXPLANATION_MODEL}" \
  --eval-model "${EVALUATION_MODEL}" \
  --num-samples "${NUM_SAMPLES}" \
  --input-path "${INPUT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --evaluate-existing-explanations \
  "$@"
