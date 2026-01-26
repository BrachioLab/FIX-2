#!/usr/bin/env bash
set -euo pipefail

python src/massmaps.py run --model gemini-2.5-flash --method socratic --run_generation --num_samples 100 "$@"
python src/massmaps.py run --model gemini-2.5-flash --method socratic --run_evaluation --num_samples 100 "$@"
