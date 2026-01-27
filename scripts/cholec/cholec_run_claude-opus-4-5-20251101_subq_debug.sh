#!/usr/bin/env bash
set -euo pipefail

python src/cholec.py run --model claude-opus-4-5-20251101 --method subq --run_generation --num_samples 5 "$@"
python src/cholec.py run --model claude-opus-4-5-20251101 --method subq --run_evaluation --num_samples 5 "$@"
