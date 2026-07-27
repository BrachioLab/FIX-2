#!/usr/bin/env python3
"""Classify when massmaps category-level claim groups need joint evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CRITERIA = (
    "Lensing Peak (Cluster) Abundance",
    "Void Size and Frequency",
    "Filament Thickness and Sharpness",
    "Fine-Scale Clumpiness",
    "Connectivity of the Cosmic Web",
    "Density Contrast Extremes",
)


FEATURE_TERMS = {
    "Lensing Peak (Cluster) Abundance": ("peak", "peaks", "red", "yellow", "overdens", "cluster", "high-significance"),
    "Void Size and Frequency": ("void", "underdens", "low-convergence", "blue", "gray", "empty", "expans"),
    "Filament Thickness and Sharpness": ("filament", "thread", "thick", "thin", "sharp", "diffuse"),
    "Fine-Scale Clumpiness": ("fine", "grain", "texture", "clump", "mini", "smooth", "homogeneous"),
    "Connectivity of the Cosmic Web": ("connect", "network", "web", "link", "isolated", "fragment", "filament"),
    "Density Contrast Extremes": ("contrast", "dense", "void", "bright", "dark", "peak", "empty"),
}

INTERPRETATION_TERMS = {
    "Lensing Peak (Cluster) Abundance": ("sigma_8", "fluctuation", "amplitude", "clumpier", "larger sigma"),
    "Void Size and Frequency": ("omega_m", "matter density", "lower omega", "reduced overall matter"),
    "Filament Thickness and Sharpness": ("sigma_8", "fluctuation", "small-scale clustering", "amplitude"),
    "Fine-Scale Clumpiness": ("sigma_8", "fluctuation", "small-scale", "amplitude"),
    "Connectivity of the Cosmic Web": ("omega_m", "matter density", "higher omega", "lower omega"),
    "Density Contrast Extremes": ("sigma_8", "variance", "density field", "fluctuation", "amplitude"),
}


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def classify_claim_group(category: str, claims: list[str], score: float | None) -> tuple[str, str]:
    n_claims = len(claims)
    if n_claims == 0:
        return "no_claims", "No claims were grouped under this criterion."
    if n_claims == 1:
        return "not_need", "Only one grouped claim; it can be judged directly."
    if score == 0.0:
        return "not_need", "The grouped claims do not align with the criterion, so joint synthesis is not needed."

    feature_terms = FEATURE_TERMS[category]
    interpretation_terms = INTERPRETATION_TERMS[category]
    feature_only = []
    interpretation_only = []
    self_contained = []
    for claim in claims:
        has_feature = has_any(claim, feature_terms)
        has_interpretation = has_any(claim, interpretation_terms)
        if has_feature and has_interpretation:
            self_contained.append(claim)
        elif has_feature:
            feature_only.append(claim)
        elif has_interpretation:
            interpretation_only.append(claim)

    if feature_only and interpretation_only:
        return (
            "need",
            "Visual evidence and parameter interpretation are split across different claims; the criterion should be judged jointly.",
        )

    if n_claims >= 4 and score == 1.0 and len(self_contained) < n_claims:
        return (
            "need",
            "Several grouped claims collectively establish the criterion-level pattern; judging them separately would lose context.",
        )

    if self_contained:
        return (
            "not_need",
            "At least one claim is already self-contained for this criterion; the remaining claims mostly repeat or add local detail.",
        )

    return (
        "borderline",
        "Multiple related claims are present, but no clear cross-claim dependency is required for the judgement.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/vanilla/massmaps_gpt-5-mini-2025-08-07.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/c2bd847_previous_pipeline/results/analysis/massmaps_joint_claim_need"),
    )
    args = parser.parse_args()

    rows = json.load(args.input.open())
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    for fallback_idx, row in enumerate(rows):
        idx = int(row.get("idx", fallback_idx))
        for category in CRITERIA:
            claims = row.get("claims_by_category", {}).get(category, []) or []
            score = row.get("category_alignment_scores", {}).get(category)
            label, judgement = classify_claim_group(category, claims, score)
            detail_rows.append(
                {
                    "idx": idx,
                    "category": category,
                    "label": label,
                    "n_claims": len(claims),
                    "category_alignment_score": score,
                    "judgement": judgement,
                    "claims": claims,
                }
            )

    summary = {
        "input": str(args.input),
        "n_examples": len(rows),
        "n_example_category_pairs": len(detail_rows),
        "rubric": {
            "need": "Joint category-level evaluation is necessary because the grouped claims must be synthesized.",
            "not_need": "A single/self-contained claim, redundant claims, or non-aligning claims can be judged without synthesis.",
            "borderline": "Grouping is useful context but not clearly necessary.",
            "no_claims": "No claims were grouped for this criterion.",
        },
        "overall_counts": Counter(row["label"] for row in detail_rows),
        "by_category": {},
    }
    for category in CRITERIA:
        category_rows = [row for row in detail_rows if row["category"] == category]
        summary["by_category"][category] = {
            "counts": Counter(row["label"] for row in category_rows),
            "examples_by_label": {
                label: sorted(row["idx"] for row in category_rows if row["label"] == label)
                for label in ("need", "not_need", "borderline", "no_claims")
            },
        }

    detail_json = args.output_dir / "massmaps_joint_claim_need_details.json"
    summary_json = args.output_dir / "massmaps_joint_claim_need_summary.json"
    detail_csv = args.output_dir / "massmaps_joint_claim_need_details.csv"

    with detail_json.open("w") as f:
        json.dump(detail_rows, f, indent=2)
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "category", "label", "n_claims", "category_alignment_score", "judgement", "claims"],
        )
        writer.writeheader()
        for row in detail_rows:
            csv_row = dict(row)
            csv_row["claims"] = " | ".join(row["claims"])
            writer.writerow(csv_row)

    print(f"saved details: {detail_json}")
    print(f"saved summary: {summary_json}")
    print(f"saved csv: {detail_csv}")
    print(json.dumps(summary["overall_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
