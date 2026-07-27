#!/usr/bin/env python3
"""Compare original, legacy, and simple legacy massmaps evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
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
            data = json.load(input_file)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        rows = data
    elif path.is_dir():
        rows = []
        for json_path in sorted(path.glob("*.json"), key=numeric_json_sort_key):
            with json_path.open("rt") as input_file:
                row = json.load(input_file)
            row.setdefault("idx", int(json_path.stem))
            rows.append(row)
    else:
        raise FileNotFoundError(path)

    by_idx = {}
    for fallback_idx, row in enumerate(rows):
        idx = int(row.get("idx", fallback_idx))
        by_idx[idx] = row
    return by_idx


def number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def extract_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    mse_loss = row.get("mse_loss") or {}
    omega_mse = number_or_none(mse_loss.get("Omega_m"))
    sigma_mse = number_or_none(mse_loss.get("sigma_8"))
    metrics = {
        "final_alignment_score": number_or_none(row.get("final_alignment_score")),
        "omega_m_mse": omega_mse,
        "sigma_8_mse": sigma_mse,
        "total_mse": None,
    }
    if omega_mse is not None and sigma_mse is not None:
        metrics["total_mse"] = omega_mse + sigma_mse
    return metrics


def default_original_path(method: str, explanation_model: str, eval_model: str) -> Path:
    with_eval = ROOT_DIR / "results" / method / f"massmaps_{safe_model_name(explanation_model)}_{safe_model_name(eval_model)}.json"
    without_eval = ROOT_DIR / "results" / method / f"massmaps_{safe_model_name(explanation_model)}.json"
    return with_eval if with_eval.exists() else without_eval


def default_legacy_dir(method: str, explanation_model: str, eval_model: str) -> Path:
    return (
        BASELINE_DIR
        / "notebooks"
        / "_dump"
        / "massmaps"
        / "final"
        / "legacy-c2bd847"
        / safe_model_name(explanation_model)
        / method
        / f"eval.{safe_model_name(eval_model)}"
    )


def default_simple_dir(method: str, explanation_model: str, eval_model: str) -> Path:
    return (
        BASELINE_DIR
        / "notebooks"
        / "_dump"
        / "massmaps"
        / "final"
        / "legacy-c2bd847-simple"
        / safe_model_name(explanation_model)
        / method
        / f"eval.simple.{safe_model_name(eval_model)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize original, legacy, and simple legacy massmaps results.")
    parser.add_argument("--method", default="vanilla")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07", help="Explanation model.")
    parser.add_argument("--eval-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--original-path", type=Path)
    parser.add_argument("--legacy-dir", type=Path)
    parser.add_argument("--simple-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--all-original", action="store_true", help="Use all original examples instead of only IDs common to all sources.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    original_path = (args.original_path or default_original_path(args.method, args.model, args.eval_model)).resolve()
    legacy_dir = (args.legacy_dir or default_legacy_dir(args.method, args.model, args.eval_model)).resolve()
    simple_dir = (args.simple_dir or default_simple_dir(args.method, args.model, args.eval_model)).resolve()

    sources = {
        "original": load_result_source(original_path),
        "legacy": load_result_source(legacy_dir),
        "legacy_simple": load_result_source(simple_dir),
    }

    if args.all_original:
        selected_ids = sorted(sources["original"])
    else:
        selected_ids = sorted(set.intersection(*(set(rows) for rows in sources.values())))

    summary: dict[str, Any] = {
        "method": args.method,
        "explanation_model": args.model,
        "eval_model": args.eval_model,
        "paths": {
            "original": str(original_path),
            "legacy": str(legacy_dir),
            "legacy_simple": str(simple_dir),
        },
        "source_counts": {name: len(rows) for name, rows in sources.items()},
        "common_count": len(selected_ids),
        "ids": selected_ids,
        "summaries": {},
    }

    metric_names = ("final_alignment_score", "omega_m_mse", "sigma_8_mse", "total_mse")
    csv_rows = []
    for source_name, rows_by_idx in sources.items():
        source_summary = {}
        for metric_name in metric_names:
            values = []
            for idx in selected_ids:
                row = rows_by_idx.get(idx)
                if row is None:
                    continue
                value = extract_metrics(row)[metric_name]
                if value is not None:
                    values.append(value)
            source_summary[metric_name] = summarize_values(values)

        summary["summaries"][source_name] = source_summary
        csv_rows.append({
            "source": source_name,
            "n": source_summary["final_alignment_score"]["n"],
            "final_alignment_mean": source_summary["final_alignment_score"]["mean"],
            "final_alignment_std": source_summary["final_alignment_score"]["std"],
            "omega_m_mse_mean": source_summary["omega_m_mse"]["mean"],
            "sigma_8_mse_mean": source_summary["sigma_8_mse"]["mean"],
            "total_mse_mean": source_summary["total_mse"]["mean"],
        })

    output_dir = BASELINE_DIR / "results" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or output_dir / f"massmaps_legacy_compare_{safe_model_name(args.model)}_{args.method}_eval-{safe_model_name(args.eval_model)}.json"
    output_csv = args.output_csv or output_dir / f"massmaps_legacy_compare_{safe_model_name(args.model)}_{args.method}_eval-{safe_model_name(args.eval_model)}.csv"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("wt") as output_file:
        json.dump(summary, output_file, indent=4)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("wt", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Compared IDs: {selected_ids}")
    print(f"Saved JSON summary: {output_json}")
    print(f"Saved CSV summary: {output_csv}")
    for row in csv_rows:
        print(
            "{source}: n={n}, final_alignment_mean={final_alignment_mean}, "
            "omega_m_mse_mean={omega_m_mse_mean}, sigma_8_mse_mean={sigma_8_mse_mean}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
