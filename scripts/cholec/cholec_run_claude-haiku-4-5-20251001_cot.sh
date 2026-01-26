#!/usr/bin/env bash
set -euo pipefail

python src/cholec.py run --model claude-haiku-4-5-20251001 --method cot --run_generation --num_samples 100 "$@"
python src/cholec.py run --model claude-haiku-4-5-20251001 --method cot --run_evaluation --num_samples 100 "$@"
