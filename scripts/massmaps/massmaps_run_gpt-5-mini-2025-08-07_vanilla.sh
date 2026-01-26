#!/usr/bin/env bash
set -euo pipefail

python src/massmaps.py run --model gpt-5-mini-2025-08-07 --method vanilla --run_generation --num_samples 100 "$@"
python src/massmaps.py run --model gpt-5-mini-2025-08-07 --method vanilla --run_evaluation --num_samples 100 "$@"
