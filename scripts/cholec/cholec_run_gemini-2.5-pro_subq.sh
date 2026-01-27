#!/usr/bin/env bash
set -euo pipefail

python src/cholec.py run --model gemini-2.5-pro --method subq --run_generation --num_samples 100 "$@"
python src/cholec.py run --model gemini-2.5-pro --method subq --run_evaluation --num_samples 100 "$@"
