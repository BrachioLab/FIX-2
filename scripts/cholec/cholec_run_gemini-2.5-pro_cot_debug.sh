#!/usr/bin/env bash
set -euo pipefail

python src/cholec.py run --model gemini-2.5-pro --method cot --run_generation --num_samples 5 "$@"
python src/cholec.py run --model gemini-2.5-pro --method cot --run_evaluation --num_samples 5 "$@"
