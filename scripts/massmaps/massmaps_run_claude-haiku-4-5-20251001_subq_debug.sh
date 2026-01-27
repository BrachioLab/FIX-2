#!/usr/bin/env bash
set -euo pipefail

python src/massmaps.py run --model claude-haiku-4-5-20251001 --method subq --run_generation --num_samples 5 "$@"
python src/massmaps.py run --model claude-haiku-4-5-20251001 --method subq --run_evaluation --num_samples 5 "$@"
