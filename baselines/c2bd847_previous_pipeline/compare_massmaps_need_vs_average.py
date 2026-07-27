#!/usr/bin/env python3
"""Compare need-label category mean against the overall category mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_COMPARISON = (
    BASELINE_DIR
    / "results"
    / "analysis"
    / "massmaps_legacy_by_joint_claim_need"
    / "massmaps_current_vs_legacy_by_joint_claim_need.json"
)
DEFAULT_OUTPUT = (
    BASELINE_DIR
    / "results"
    / "analysis"
    / "massmaps_legacy_by_joint_claim_need"
    / "massmaps_need_vs_average_category_mean.json"
)


def weighted_average(comparison: dict, method_prefix: str) -> float:
    total_score = 0.0
    total_n = 0
    for row in comparison.values():
        mean_key = f"{method_prefix}_matching_category_mean"
        n_key = f"{method_prefix}_matching_category_n"
        mean = row[mean_key]
        n = row[n_key]
        total_score += mean * n
        total_n += n
    return total_score / total_n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.comparison_path.open("r") as f:
        comparison = json.load(f)["comparison"]

    current_average = weighted_average(comparison, "current")
    legacy_average = weighted_average(comparison, "legacy")
    current_need = comparison["need"]["current_matching_category_mean"]
    legacy_need = comparison["need"]["legacy_matching_category_mean"]

    result = {
        "comparison_path": str(args.comparison_path),
        "current": {
            "need_category_mean": current_need,
            "overall_category_mean": current_average,
            "need_minus_average": current_need - current_average,
        },
        "legacy": {
            "need_category_mean": legacy_need,
            "overall_category_mean": legacy_average,
            "need_minus_average": legacy_need - legacy_average,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"saved: {args.output_path}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
