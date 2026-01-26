#!/usr/bin/env bash
set -euo pipefail

python src/cholec.py run --model claude-opus-4-5-20251101 --method socratic --run_generation --num_samples 100 "$@"
python src/cholec.py run --model claude-opus-4-5-20251101 --method socratic --run_evaluation --num_samples 100 "$@"
