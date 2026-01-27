#!/usr/bin/env bash
set -euo pipefail

python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method socratic --run_generation --num_samples 100 "$@"
python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method socratic --run_evaluation --num_samples 100 "$@"
