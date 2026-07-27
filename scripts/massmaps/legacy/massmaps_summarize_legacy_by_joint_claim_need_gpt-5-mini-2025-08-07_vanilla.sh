#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" baselines/c2bd847_previous_pipeline/summarize_massmaps_legacy_by_joint_need.py "$@"
