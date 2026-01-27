#!/usr/bin/env bash
set -euo pipefail

python src/massmaps.py run --model gemini-2.5-flash --method vanilla --run_generation --num_samples 5 "$@"
python src/massmaps.py run --model gemini-2.5-flash --method vanilla --run_evaluation --num_samples 5 "$@"
