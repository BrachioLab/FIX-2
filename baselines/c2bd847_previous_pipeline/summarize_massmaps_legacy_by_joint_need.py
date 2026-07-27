#!/usr/bin/env python3
"""Summarize legacy massmaps scores by joint-claim-need labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = BASELINE_DIR / "results" / "analysis" / "massmaps_joint_claim_need" / "massmaps_joint_claim_need_details.json"
DEFAULT_LEGACY_DIR = (
    BASELINE_DIR
    / "notebooks"
    / "_dump"
    / "massmaps"
    / "final"
    / "legacy-c2bd847"
    / "gpt-5-mini-2025-08-07"
    / "vanilla"
    / "eval.gpt-5-mini-2025-08-07"
)
DEFAULT_OUTPUT_DIR = BASELINE_DIR / "results" / "analysis" / "massmaps_legacy_by_joint_claim_need"
LABELS = ("need", "not_need", "borderline", "no_claims")


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def legacy_category_score(row: dict[str, Any], category: str) -> float | None:
    scores = row.get("alignment_scores") or []
    categories = row.get("alignment_categories") or []
    matched = [number(score) for score, cat in zip(scores, categories) if cat == category]
    matched = [score for score in matched if score is not None]
    if not matched:
        return None
    return mean(matched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    label_rows = load_json(args.labels_path)
    if not isinstance(label_rows, list):
        raise ValueError(f"Expected JSON list in {args.labels_path}")

    legacy_by_idx: dict[int, dict[str, Any]] = {}
    for path in sorted(args.legacy_dir.glob("*.json")):
        row = load_json(path)
        idx = int(row.get("idx", path.stem))
        legacy_by_idx[idx] = row
    missing = sorted({int(row["idx"]) for row in label_rows} - set(legacy_by_idx))
    if missing:
        raise FileNotFoundError(f"Missing legacy outputs for idx values: {missing[:20]}")

    detail_rows = []
    example_final_by_label: dict[str, list[float]] = defaultdict(list)
    category_score_by_label: dict[str, list[float]] = defaultdict(list)

    for label_row in label_rows:
        idx = int(label_row["idx"])
        label = label_row["label"]
        category = label_row["category"]
        legacy_row = legacy_by_idx[idx]
        example_score = number(legacy_row.get("final_alignment_score"))
        category_score = legacy_category_score(legacy_row, category)

        if example_score is not None:
            example_final_by_label[label].append(example_score)
        if category_score is not None:
            category_score_by_label[label].append(category_score)

        detail_rows.append(
            {
                "idx": idx,
                "category": category,
                "label": label,
                "legacy_example_final_alignment_score": example_score,
                "legacy_matching_category_claim_score": category_score,
                "n_current_grouped_claims": label_row.get("n_claims"),
                "current_category_alignment_score": label_row.get("category_alignment_score"),
                "judgement": label_row.get("judgement"),
            }
        )

    summary = {
        "labels_path": str(args.labels_path),
        "legacy_dir": str(args.legacy_dir),
        "n_legacy_examples": len(legacy_by_idx),
        "n_example_category_pairs": len(detail_rows),
        "by_label": {
            label: {
                "n_pairs": sum(1 for row in detail_rows if row["label"] == label),
                "legacy_example_final_alignment_score": stats(example_final_by_label[label]),
                "legacy_matching_category_claim_score": stats(category_score_by_label[label]),
            }
            for label in LABELS
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "massmaps_legacy_by_joint_claim_need_summary.json"
    details_path = args.output_dir / "massmaps_legacy_by_joint_claim_need_details.json"
    csv_path = args.output_dir / "massmaps_legacy_by_joint_claim_need_details.csv"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    with details_path.open("w") as f:
        json.dump(detail_rows, f, indent=2)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"saved summary: {summary_path}")
    print(f"saved details: {details_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(summary["by_label"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
