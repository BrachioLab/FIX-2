#!/usr/bin/env bash
set -euo pipefail

python src/massmaps.py run --model gpt-5-mini-2025-08-07 --method cot --run_evaluation --num_samples 100 "$@"
