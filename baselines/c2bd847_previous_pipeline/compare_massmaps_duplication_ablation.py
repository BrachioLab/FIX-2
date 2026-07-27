#!/usr/bin/env python3
"""Compare original massmaps scores against a duplication ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent


def safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def numeric_json_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def load_result_source(path: Path) -> dict[int, dict[str, Any]]:
    if path.is_file():
        with path.open("rt") as input_file:
            rows = json.load(input_file)
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON list in {path}")
    elif path.is_dir():
        rows = []
        for json_path in sorted(path.glob("*.json"), key=numeric_json_sort_key):
            with json_path.open("rt") as input_file:
                row = json.load(input_file)
            row.setdefault("idx", int(json_path.stem))
            rows.append(row)
    else:
        raise FileNotFoundError(path)

    return {int(row.get("idx", fallback_idx)): row for fallback_idx, row in enumerate(rows)}


def score(row: dict[str, Any]) -> float:
    return float(row["final_alignment_score"])


def default_original_current_path(method: str, model: str, eval_model: str) -> Path:
    with_eval = ROOT_DIR / "results" / method / f"massmaps_{safe_model_name(model)}_{safe_model_name(eval_model)}.json"
    without_eval = ROOT_DIR / "results" / method / f"massmaps_{safe_model_name(model)}.json"
    return with_eval if with_eval.exists() else without_eval


def default_original_legacy_dir(method: str, model: str, eval_model: str) -> Path:
    return (
        BASELINE_DIR
        / "notebooks"
        / "_dump"
        / "massmaps"
        / "final"
        / "legacy-c2bd847"
        / safe_model_name(model)
        / method
        / f"eval.{safe_model_name(eval_model)}"
    )


def default_original_simple_dir(method: str, model: str, eval_model: str) -> Path:
    return (
        BASELINE_DIR
        / "notebooks"
        / "_dump"
        / "massmaps"
        / "final"
        / "legacy-c2bd847-simple"
        / safe_model_name(model)
        / method
        / f"eval.simple.{safe_model_name(eval_model)}"
    )


def default_ablation_dir(evaluator: str, variant: str, method: str, model: str, eval_model: str) -> Path:
    eval_dir = f"eval.{safe_model_name(eval_model)}"
    if evaluator == "legacy_simple":
        eval_dir = f"eval.simple.{safe_model_name(eval_model)}"
    return (
        BASELINE_DIR
        / "results"
        / "duplication_ablation"
        / "evaluations"
        / evaluator
        / variant
        / safe_model_name(model)
        / method
        / eval_dir
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original vs duplication ablation massmaps scores.")
    parser.add_argument("--variant", choices=("duplicate_high", "duplicate_low"), required=True)
    parser.add_argument("--method", default="vanilla")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--eval-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_sources = {
        "current": default_original_current_path(args.method, args.model, args.eval_model),
        "legacy": default_original_legacy_dir(args.method, args.model, args.eval_model),
        "legacy_simple": default_original_simple_dir(args.method, args.model, args.eval_model),
    }
    ablation_sources = {
        evaluator: default_ablation_dir(evaluator, args.variant, args.method, args.model, args.eval_model)
        for evaluator in original_sources
    }

    rows = []
    summary = {
        "variant": args.variant,
        "method": args.method,
        "model": args.model,
        "eval_model": args.eval_model,
        "paths": {
            evaluator: {
                "original": str(original_sources[evaluator]),
                "ablation": str(ablation_sources[evaluator]),
            }
            for evaluator in original_sources
        },
        "evaluators": {},
    }

    for evaluator in ("current", "legacy", "legacy_simple"):
        original = load_result_source(original_sources[evaluator])
        ablation = load_result_source(ablation_sources[evaluator])
        ids = sorted(set(original) & set(ablation))
        original_scores = []
        ablation_scores = []

        for idx in ids:
            original_score = score(original[idx])
            ablation_score = score(ablation[idx])
            original_scores.append(original_score)
            ablation_scores.append(ablation_score)
            rows.append({
                "evaluator": evaluator,
                "idx": idx,
                "original_score": original_score,
                "ablation_score": ablation_score,
                "delta": ablation_score - original_score,
            })

        original_mean = mean(original_scores) if original_scores else None
        ablation_mean = mean(ablation_scores) if ablation_scores else None
        summary["evaluators"][evaluator] = {
            "ids": ids,
            "n": len(ids),
            "original_mean": original_mean,
            "ablation_mean": ablation_mean,
            "delta_mean": None if original_mean is None or ablation_mean is None else ablation_mean - original_mean,
            "examples": [row for row in rows if row["evaluator"] == evaluator],
        }

    output_dir = BASELINE_DIR / "results" / "duplication_ablation" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or output_dir / f"massmaps_{args.variant}_{safe_model_name(args.model)}_{args.method}_eval-{safe_model_name(args.eval_model)}.json"
    output_csv = args.output_csv or output_dir / f"massmaps_{args.variant}_{safe_model_name(args.model)}_{args.method}_eval-{safe_model_name(args.eval_model)}.csv"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("wt") as output_file:
        json.dump(summary, output_file, indent=4)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("wt", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["evaluator", "idx", "original_score", "ablation_score", "delta"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV: {output_csv}")
    for evaluator, evaluator_summary in summary["evaluators"].items():
        print(
            f"{evaluator}: n={evaluator_summary['n']}, "
            f"original_mean={evaluator_summary['original_mean']}, "
            f"ablation_mean={evaluator_summary['ablation_mean']}, "
            f"delta_mean={evaluator_summary['delta_mean']}"
        )
        for row in evaluator_summary["examples"]:
            print(
                f"  idx={row['idx']}: original={row['original_score']}, "
                f"ablation={row['ablation_score']}, delta={row['delta']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
