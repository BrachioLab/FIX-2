#!/usr/bin/env bash
set -euo pipefail

python src/cholec.py run --model gpt-5.2-pro-2025-12-11 --method cot --run_generation --num_samples 5 "$@"
python src/cholec.py run --model gpt-5.2-pro-2025-12-11 --method cot --run_evaluation --num_samples 5 "$@"
