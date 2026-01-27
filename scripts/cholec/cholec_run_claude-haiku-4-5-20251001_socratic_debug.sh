#!/usr/bin/env bash
set -euo pipefail

python src/cholec.py run --model claude-haiku-4-5-20251001 --method socratic --run_generation --num_samples 5 "$@"
python src/cholec.py run --model claude-haiku-4-5-20251001 --method socratic --run_evaluation --num_samples 5 "$@"
