#!/usr/bin/env python3
"""Analyze whether category-level claim groups need joint evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("massmaps", "cholec", "cardiac", "sepsis", "supernova", "emotion", "politeness")
MODEL = "gpt-5-mini-2025-08-07"

OBSERVATION_TERMS = (
    "shows",
    "show",
    "contains",
    "has",
    "there is",
    "there are",
    "visible",
    "appears",
    "observed",
    "present",
    "absence",
    "no ",
    "not ",
    "rate",
    "count",
    "value",
    "level",
    "score",
    "region",
    "pattern",
    "trace",
    "curve",
    "text",
    "message",
    "patient",
)

INTERPRETATION_TERMS = (
    "indicates",
    "suggests",
    "consistent with",
    "therefore",
    "because",
    "supports",
    "points to",
    "implies",
    "reflects",
    "rules out",
    "ruling out",
    "risk",
    "prediction",
    "classification",
    "label",
    "diagnosis",
    "safe",
    "unsafe",
    "polite",
    "rude",
    "emotion",
    "sigma",
    "omega",
)

MASSMAPS_FEATURE_TERMS = {
    "Lensing Peak (Cluster) Abundance": ("peak", "peaks", "red", "yellow", "overdens", "cluster", "high-significance"),
    "Void Size and Frequency": ("void", "underdens", "low-convergence", "blue", "gray", "empty", "expans"),
    "Filament Thickness and Sharpness": ("filament", "thread", "thick", "thin", "sharp", "diffuse"),
    "Fine-Scale Clumpiness": ("fine", "grain", "texture", "clump", "mini", "smooth", "homogeneous"),
    "Connectivity of the Cosmic Web": ("connect", "network", "web", "link", "isolated", "fragment", "filament"),
    "Density Contrast Extremes": ("contrast", "dense", "void", "bright", "dark", "peak", "empty"),
}

MASSMAPS_INTERPRETATION_TERMS = {
    "Lensing Peak (Cluster) Abundance": ("sigma_8", "fluctuation", "amplitude", "clumpier", "larger sigma"),
    "Void Size and Frequency": ("omega_m", "matter density", "lower omega", "reduced overall matter"),
    "Filament Thickness and Sharpness": ("sigma_8", "fluctuation", "small-scale clustering", "amplitude"),
    "Fine-Scale Clumpiness": ("sigma_8", "fluctuation", "small-scale", "amplitude"),
    "Connectivity of the Cosmic Web": ("omega_m", "matter density", "higher omega", "lower omega"),
    "Density Contrast Extremes": ("sigma_8", "variance", "density field", "fluctuation", "amplitude"),
}


RUBRIC_TEXT = """# Joint Claim Need Judgement Rubric

This analysis was performed by Codex over saved result JSON files, without making new model API calls.

Internal judgement prompt/rubric used:

You are analyzing a saved explanation-evaluation result. For each dataset example and each expert criterion/category, you are given:
- the dataset name,
- the criterion/category name,
- the claims grouped under that criterion,
- the existing category alignment score, when available,
- the existing category alignment reasoning, when available.

Decide whether evaluating that criterion actually requires considering multiple claims together.

Use these labels:
- need: The criterion judgement depends on synthesizing multiple grouped claims. Use this when one claim supplies evidence and another supplies interpretation, when multiple claims cover distinct required facets of the criterion, or when the criterion-level score would be misleading if the claims were judged independently.
- not_need: The criterion can be judged from one self-contained claim, the grouped claims are redundant variants of one point, or the grouped claims do not align with the criterion.
- borderline: Multiple related claims are present and grouping is useful context, but there is no clear dependency that makes joint evaluation necessary.
- no_claims: No claims were grouped under the criterion.

Tie-breaking rules:
- Empty groups are no_claims.
- Single-claim groups are not_need.
- Groups with alignment score 0 are not_need unless the reasoning explicitly says several claims jointly contradict the criterion.
- Groups with two or more claims are need when evidence and interpretation are split across claims, or when the category name itself contains multiple facets and the claims cover different facets.
- Otherwise, use borderline for multi-claim groups where grouping helps but necessity is unclear.

The script encodes this rubric deterministically so the result can be rerun later.

For massmaps, the saved labels preserve the earlier massmaps-specific Codex judgement. That pass used the same labels, with category-specific visual-evidence and cosmological-interpretation term checks for the six massmaps criteria.
"""


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2)


def normalize_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def category_has_multiple_facets(category: str) -> bool:
    lowered = category.lower()
    return any(marker in lowered for marker in (" and ", " or ", "/", "&", "(", ")"))


def claims_cover_different_facets(category: str, claims: list[str]) -> bool:
    category_terms = token_set(category)
    category_terms = {t for t in category_terms if len(t) >= 4}
    if len(category_terms) < 2:
        return False
    covered_by_claim = [token_set(claim) & category_terms for claim in claims]
    nonempty = [terms for terms in covered_by_claim if terms]
    if len(nonempty) < 2:
        return False
    union = set().union(*nonempty)
    return len(union) >= 2 and any(a != b for a in nonempty for b in nonempty)


def classify_group(category: str, claims: list[str], score: float | None, reasoning: str | None) -> tuple[str, str]:
    if not claims:
        return "no_claims", "No claims were grouped under this criterion."
    if len(claims) == 1:
        return "not_need", "Only one grouped claim; it can be judged directly."
    if score == 0.0:
        return "not_need", "The grouped claims have zero category alignment, so synthesis is not needed to credit the criterion."

    has_observation_only = False
    has_interpretation_only = False
    has_self_contained = False
    for claim in claims:
        has_observation = has_any(claim, OBSERVATION_TERMS)
        has_interpretation = has_any(claim, INTERPRETATION_TERMS)
        if has_observation and has_interpretation:
            has_self_contained = True
        elif has_observation:
            has_observation_only = True
        elif has_interpretation:
            has_interpretation_only = True

    if has_observation_only and has_interpretation_only:
        return "need", "Evidence and interpretation are split across different claims."

    if category_has_multiple_facets(category) and claims_cover_different_facets(category, claims):
        return "need", "The criterion has multiple facets and different claims cover different facets."

    reasoning_text = reasoning or ""
    if len(claims) >= 3 and score == 1.0 and has_any(reasoning_text, ("claims", "together", "combination", "collectively", "multiple")):
        return "need", "The saved reasoning describes the grouped claims collectively supporting the criterion."

    if len(claims) >= 4 and score == 1.0 and not has_self_contained:
        return "need", "Several high-alignment claims collectively establish the criterion-level judgement."

    if has_self_contained:
        return "not_need", "At least one grouped claim is already self-contained; other claims mainly add detail or repetition."

    return "borderline", "Multiple related claims are present, but joint evaluation is useful rather than clearly necessary."


def classify_massmaps_group(category: str, claims: list[str], score: float | None) -> tuple[str, str]:
    if not claims:
        return "no_claims", "No claims were grouped under this criterion."
    if len(claims) == 1:
        return "not_need", "Only one grouped claim; it can be judged directly."
    if score == 0.0:
        return "not_need", "The grouped claims do not align with the criterion, so joint synthesis is not needed."

    feature_only = []
    interpretation_only = []
    self_contained = []
    for claim in claims:
        has_feature = has_any(claim, MASSMAPS_FEATURE_TERMS[category])
        has_interpretation = has_any(claim, MASSMAPS_INTERPRETATION_TERMS[category])
        if has_feature and has_interpretation:
            self_contained.append(claim)
        elif has_feature:
            feature_only.append(claim)
        elif has_interpretation:
            interpretation_only.append(claim)

    if feature_only and interpretation_only:
        return "need", "Visual evidence and parameter interpretation are split across different claims; the criterion should be judged jointly."

    if len(claims) >= 4 and score == 1.0 and len(self_contained) < len(claims):
        return "need", "Several grouped claims collectively establish the criterion-level pattern; judging them separately would lose context."

    if self_contained:
        return "not_need", "At least one claim is already self-contained for this criterion; the remaining claims mostly repeat or add local detail."

    return "borderline", "Multiple related claims are present, but no clear cross-claim dependency is required for the judgement."


def example_id(row: dict[str, Any], fallback_idx: int) -> str:
    if "idx" in row:
        return str(row["idx"])
    if "_filename" in row:
        return str(row["_filename"])
    if "id" in row:
        return str(row["id"])
    if "record_name" in row:
        return str(row["record_name"])
    return str(fallback_idx)


def analyze_dataset(dataset: str, input_path: Path) -> list[dict[str, Any]]:
    rows = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON list in {input_path}")

    detail_rows: list[dict[str, Any]] = []
    for fallback_idx, row in enumerate(rows):
        claims_by_category = row.get("claims_by_category", {}) or {}
        scores = row.get("category_alignment_scores", {}) or {}
        reasonings = row.get("category_alignment_reasonings", {}) or {}
        for category, claims in claims_by_category.items():
            claims = claims or []
            score = normalize_score(scores.get(category))
            if dataset == "massmaps":
                label, judgement = classify_massmaps_group(category, claims, score)
            else:
                label, judgement = classify_group(category, claims, score, reasonings.get(category))
            detail_rows.append(
                {
                    "dataset": dataset,
                    "source_path": str(input_path),
                    "example_position": fallback_idx,
                    "example_id": example_id(row, fallback_idx),
                    "category": category,
                    "label": label,
                    "n_claims": len(claims),
                    "category_alignment_score": score,
                    "judgement": judgement,
                    "category_alignment_reasoning": reasonings.get(category),
                    "claims": claims,
                }
            )
    return detail_rows


def summarize(detail_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall_counts = Counter(row["label"] for row in detail_rows)
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        ds_rows = [row for row in detail_rows if row["dataset"] == dataset]
        if not ds_rows:
            continue
        ds_counts = Counter(row["label"] for row in ds_rows)
        by_category = {}
        for category in sorted({row["category"] for row in ds_rows}):
            cat_rows = [row for row in ds_rows if row["category"] == category]
            cat_counts = Counter(row["label"] for row in cat_rows)
            by_category[category] = {
                "counts": dict(cat_counts),
                "percent_need": 100.0 * cat_counts.get("need", 0) / len(cat_rows),
                "n": len(cat_rows),
            }
        by_dataset[dataset] = {
            "n_example_category_pairs": len(ds_rows),
            "counts": dict(ds_counts),
            "percent_need": 100.0 * ds_counts.get("need", 0) / len(ds_rows),
            "mean_claims_per_pair": mean(row["n_claims"] for row in ds_rows),
            "by_category": by_category,
        }

    return {
        "model": MODEL,
        "method": "vanilla",
        "n_example_category_pairs": len(detail_rows),
        "overall_counts": dict(overall_counts),
        "overall_percent_need": 100.0 * overall_counts.get("need", 0) / len(detail_rows),
        "by_dataset": by_dataset,
        "rubric_file": "joint_claim_need_rubric.md",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "source_path",
        "example_position",
        "example_id",
        "category",
        "label",
        "n_claims",
        "category_alignment_score",
        "judgement",
        "category_alignment_reasoning",
        "claims",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["claims"] = " | ".join(str(claim) for claim in row["claims"])
            writer.writerow(csv_row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/vanilla"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/c2bd847_previous_pipeline/results/analysis/joint_claim_need_all_datasets"),
    )
    args = parser.parse_args()

    detail_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        input_path = args.results_dir / f"{dataset}_{MODEL}.json"
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        detail_rows.extend(analyze_dataset(dataset, input_path))

    summary = summarize(detail_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "joint_claim_need_details.json", detail_rows)
    dump_json(args.output_dir / "joint_claim_need_summary.json", summary)
    write_csv(args.output_dir / "joint_claim_need_details.csv", detail_rows)
    (args.output_dir / "joint_claim_need_rubric.md").write_text(RUBRIC_TEXT)

    print(f"saved details: {args.output_dir / 'joint_claim_need_details.json'}")
    print(f"saved summary: {args.output_dir / 'joint_claim_need_summary.json'}")
    print(f"saved csv: {args.output_dir / 'joint_claim_need_details.csv'}")
    print(f"saved rubric: {args.output_dir / 'joint_claim_need_rubric.md'}")
    print(json.dumps(summary["overall_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
