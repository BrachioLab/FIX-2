#!/usr/bin/env python3
"""Run massmaps duplicate-claim ablations.

The ablation appends a paraphrase of one already-extracted claim to the
explanation text used for evaluation. The source result JSON is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


BASELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASELINE_DIR.parents[1]
SRC_DIR = ROOT_DIR / "src"

DUPLICATION_ABLATION_PROMPT = """You will be given:
- The original explanation text for a weak lensing mass map prediction.
- A list of atomic claims extracted from that explanation.
- One selected atomic claim from the list.

Your task is to write a faithful paraphrase of the selected claim.

Requirements:
- Preserve the meaning, parameter direction, uncertainty, and visual details of the selected claim.
- Make the paraphrase standalone and understandable without referring back to the original explanation.
- Do not add new evidence, categories, or conclusions.
- Do not copy the selected claim verbatim.
- Output exactly two lines in the format shown below.

Here are some examples:

Example 1:
ORIGINAL TEXT: The map contains many blue and gray regions, suggesting underdense areas. This supports a moderate Omega_m estimate.
ATOMIC CLAIMS:
The map contains many blue and gray regions.
Blue and gray regions indicate underdense areas in the map.
The underdense areas support a moderate Omega_m estimate.
SELECTED CLAIM: The underdense areas support a moderate Omega_m estimate.

OUTPUT:
ORIGINAL: The underdense areas support a moderate Omega_m estimate.
PARAPHRASE: The presence of underdense regions in the map is consistent with estimating Omega_m at a moderate level.

Example 2:
ORIGINAL TEXT: Several yellow peaks appear in the weak lensing map, indicating compact high-convergence structures and a relatively high sigma_8.
ATOMIC CLAIMS:
Several yellow peaks appear in the weak lensing map.
Yellow peaks indicate compact high-convergence structures.
Compact high-convergence structures indicate a relatively high sigma_8.
SELECTED CLAIM: Compact high-convergence structures indicate a relatively high sigma_8.

OUTPUT:
ORIGINAL: Compact high-convergence structures indicate a relatively high sigma_8.
PARAPHRASE: The compact high-convergence features in the map point toward a comparatively elevated sigma_8 value.

Now write a faithful paraphrase for the selected claim.

ORIGINAL TEXT: {original_text}
ATOMIC CLAIMS:
{atomic_claims}
SELECTED CLAIM: {selected_claim}

OUTPUT:
"""


def safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def load_json(path: Path) -> Any:
    with path.open("rt") as input_file:
        return json.load(input_file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt") as output_file:
        json.dump(data, output_file, indent=4)


def default_source_path(method: str, model: str, eval_model: str) -> Path:
    with_eval = ROOT_DIR / "results" / method / f"massmaps_{safe_model_name(model)}_{safe_model_name(eval_model)}.json"
    without_eval = ROOT_DIR / "results" / method / f"massmaps_{safe_model_name(model)}.json"
    return with_eval if with_eval.exists() else without_eval


def score_claims(row: dict[str, Any]) -> list[dict[str, Any]]:
    claims = row.get("relevant_claims") or row.get("claims") or []
    claims_by_category = row.get("claims_by_category") or {}
    category_scores = row.get("category_alignment_scores") or {}

    scored = []
    for claim in claims:
        scores = []
        categories = []
        for category, category_claims in claims_by_category.items():
            if claim in category_claims:
                scores.append(float(category_scores.get(category, 0.0)))
                categories.append(category)
        scored.append({
            "claim": claim,
            "score": max(scores) if scores else 0.0,
            "categories": categories,
        })
    return scored


def choose_claim(row: dict[str, Any], variant: str) -> dict[str, Any]:
    scored = score_claims(row)
    if not scored:
        claims = row.get("claims") or []
        if not claims:
            raise ValueError(f"Row idx={row.get('idx')} has no claims to duplicate")
        return {"claim": claims[0], "score": 0.0, "categories": []}
    if variant.endswith("_high"):
        return max(scored, key=lambda item: item["score"])
    if variant.endswith("_low"):
        return min(scored, key=lambda item: item["score"])
    raise ValueError(f"Unknown ablation variant: {variant}")


def parse_paraphrase(raw_output: str, selected_claim: str) -> str:
    match = re.search(r"(?im)^\s*PARAPHRASE\s*:\s*(.+?)\s*$", raw_output or "")
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in (raw_output or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.lower().startswith("original:"):
            return re.sub(r"(?i)^paraphrase\s*:\s*", "", line).strip()
    return selected_claim


def load_llm(eval_model: str):
    sys.path.insert(0, str(SRC_DIR))
    from llms import load_model

    return load_model(eval_model)


def ablation_variants(args: argparse.Namespace) -> tuple[str, str]:
    if args.variant:
        return (args.variant,)
    if args.exact_duplicate:
        return ("duplicate_exact_high", "duplicate_exact_low")
    return ("duplicate_high", "duplicate_low")


def ablation_source_path(args: argparse.Namespace, variant: str) -> Path:
    suffix = "exact_duplicate" if args.exact_duplicate else "paraphrased"
    return (args.ablation_dir / variant / f"massmaps_{safe_model_name(args.model)}_{args.method}_{suffix}.json").resolve()


def generate_ablation_sources(args: argparse.Namespace) -> list[Path]:
    source_path = (args.source_path or default_source_path(args.method, args.model, args.eval_model)).resolve()
    source_rows = load_json(source_path)
    if not isinstance(source_rows, list):
        raise ValueError(f"Expected JSON list in {source_path}")
    if args.num_samples > 0:
        source_rows = source_rows[: args.num_samples]

    llm = None if args.exact_duplicate else load_llm(args.paraphrase_model)
    output_paths = []
    for variant in ablation_variants(args):
        output_path = ablation_source_path(args, variant)
        generated_dir = (
            args.generated_dir
            / variant
            / safe_model_name(args.model)
            / args.method
            / (f"exact.{safe_model_name(args.paraphrase_model)}" if args.exact_duplicate else f"paraphrase.{safe_model_name(args.paraphrase_model)}")
        ).resolve()

        ablated_rows = []
        for fallback_idx, row in enumerate(source_rows):
            row_copy = deepcopy(row)
            example_idx = int(row_copy.get("idx", fallback_idx))
            record_path = generated_dir / f"{example_idx}.json"

            if record_path.exists() and not args.overwrite:
                generation_record = load_json(record_path)
            else:
                selected = choose_claim(row_copy, variant)
                atomic_claims = "\n".join(row_copy.get("claims") or [])
                prompt = DUPLICATION_ABLATION_PROMPT.format(
                    original_text=row_copy.get("llm_explanation", ""),
                    atomic_claims=atomic_claims,
                    selected_claim=selected["claim"],
                )
                if args.exact_duplicate:
                    raw_output = f"ORIGINAL: {selected['claim']}\nPARAPHRASE: {selected['claim']}"
                    paraphrase = selected["claim"]
                else:
                    raw_output = llm(prompt)
                    paraphrase = parse_paraphrase(raw_output, selected["claim"])
                original_explanation = row_copy.get("llm_explanation", "")
                ablated_explanation = f"{original_explanation.rstrip()}\n\n{paraphrase}"

                generation_record = {
                    "idx": example_idx,
                    "variant": variant,
                    "source_path": str(source_path),
                    "explanation_model": args.model,
                    "evaluation_model": args.eval_model,
                    "generation_method": args.method,
                    "paraphrase_model": args.paraphrase_model,
                    "exact_duplicate": args.exact_duplicate,
                    "original_explanation": original_explanation,
                    "atomic_claims": row_copy.get("claims") or [],
                    "relevant_claims": row_copy.get("relevant_claims") or [],
                    "scored_claims": score_claims(row_copy),
                    "selected_claim": selected["claim"],
                    "selected_claim_score": selected["score"],
                    "selected_claim_categories": selected["categories"],
                    "paraphrase_prompt": prompt,
                    "paraphrase_raw_output": raw_output,
                    "paraphrased_claim": paraphrase,
                    "ablated_explanation": ablated_explanation,
                }
                write_json(record_path, generation_record)

            row_copy["llm_explanation_original"] = generation_record["original_explanation"]
            row_copy["llm_explanation"] = generation_record["ablated_explanation"]
            row_copy["duplication_ablation"] = {
                "variant": variant,
                "selected_claim": generation_record["selected_claim"],
                "selected_claim_score": generation_record["selected_claim_score"],
                "selected_claim_categories": generation_record["selected_claim_categories"],
                "exact_duplicate": generation_record.get("exact_duplicate", args.exact_duplicate),
                "paraphrase_model": generation_record["paraphrase_model"],
                "paraphrase_raw_output": generation_record["paraphrase_raw_output"],
                "paraphrased_claim": generation_record["paraphrased_claim"],
            }
            for key in [
                "claims",
                "relevant_claims",
                "claims_by_category",
                "category_alignment_scores",
                "category_alignment_reasonings",
                "alignment_scores",
                "alignment_categories",
                "alignment_reasonings",
                "final_alignment_score",
            ]:
                row_copy.pop(key, None)
            ablated_rows.append(row_copy)

        write_json(output_path, ablated_rows)
        output_paths.append(output_path)
        print(f"saved ablation source: {output_path}")
        print(f"saved ablation generation records under: {generated_dir}")

    return output_paths


def evaluate_current(args: argparse.Namespace, input_path: Path, variant: str) -> Path:
    sys.path.insert(0, str(SRC_DIR))
    import torch
    from tqdm.auto import tqdm
    import massmaps

    rows = load_json(input_path)
    eval_model_obj = massmaps.load_model(args.eval_model)
    output_dir = (
        args.output_dir
        / "current"
        / variant
        / safe_model_name(args.model)
        / args.method
        / f"eval.{safe_model_name(args.eval_model)}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for fallback_idx, row in enumerate(tqdm(rows, desc=f"current-{variant}")):
        idx = int(row.get("idx", fallback_idx))
        save_path = output_dir / f"{idx}.json"
        if save_path.exists() and not args.overwrite:
            print(f"exists, skipping: {save_path}")
            continue

        row_copy = deepcopy(row)
        if not isinstance(row_copy["input"], torch.Tensor):
            row_copy["input"] = torch.tensor(row_copy["input"])
        example = massmaps.MassMapsExample(
            input=row_copy["input"],
            answer=row_copy["answer"],
            llm_answer=row_copy["llm_answer"],
            llm_explanation=row_copy["llm_explanation"],
        )
        example.__dict__ = row_copy
        example.idx = idx

        claims = massmaps.isolate_individual_features(example.llm_explanation, model=eval_model_obj)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]
        example.relevant_claims = massmaps.distill_relevant_features(
            example.input,
            example.llm_answer,
            example.claims,
            model=eval_model_obj,
            verbose=args.verbose,
        )
        claims_by_category, category_alignment_scores, category_alignment_reasonings = massmaps.calculate_expert_alignment_score(
            example.relevant_claims,
            eval_model_obj,
        )
        example.claims_by_category = claims_by_category
        example.category_alignment_scores = category_alignment_scores
        example.category_alignment_reasonings = category_alignment_reasonings
        alignment_matrix = massmaps.make_alignment_matrix(
            example.claims,
            claims_by_category,
            category_alignment_scores,
        )
        example.final_alignment_score = float(alignment_matrix.max(axis=-1).mean())

        save_dict = {}
        for key, value in example.__dict__.items():
            save_dict[key] = value if not isinstance(value, torch.Tensor) else value.cpu().numpy().tolist()
        write_json(save_path, save_dict)

    return output_dir


def run_legacy_wrapper(args: argparse.Namespace, input_path: Path, variant: str, simple: bool) -> Path:
    label = "legacy_simple" if simple else "legacy"
    output_dir = (
        args.output_dir
        / label
        / variant
        / safe_model_name(args.model)
        / args.method
        / (f"eval.simple.{safe_model_name(args.eval_model)}" if simple else f"eval.{safe_model_name(args.eval_model)}")
    ).resolve()
    cmd = [
        sys.executable,
        str(BASELINE_DIR / "run_baseline.py"),
        "--dataset",
        "massmaps",
        "--method",
        args.method,
        "--model",
        args.model,
        "--eval-model",
        args.eval_model,
        "--num-samples",
        "0",
        "--input-path",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--simple-baseline" if simple else "--evaluate-existing-explanations",
    ]
    if args.overwrite:
        print("overwrite requested for source/current; legacy wrappers still skip existing files by design")
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run massmaps duplicate-claim ablation.")
    parser.add_argument("--method", default="vanilla")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07", help="Explanation model.")
    parser.add_argument("--eval-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--paraphrase-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--ablation-dir", type=Path, default=BASELINE_DIR / "results" / "duplication_ablation" / "sources")
    parser.add_argument("--generated-dir", type=Path, default=BASELINE_DIR / "results" / "duplication_ablation" / "generated")
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR / "results" / "duplication_ablation" / "evaluations")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--exact-duplicate", action="store_true", help="Append the selected claim verbatim instead of paraphrasing it.")
    parser.add_argument(
        "--variant",
        choices=("duplicate_high", "duplicate_low", "duplicate_exact_high", "duplicate_exact_low"),
        help="Restrict generation/evaluation to one duplication variant.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.ablation_dir = args.ablation_dir.resolve()
    args.generated_dir = args.generated_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.evaluate_only:
        source_paths = [ablation_source_path(args, variant) for variant in ablation_variants(args)]
    else:
        source_paths = generate_ablation_sources(args)

    if args.generate_only:
        return 0

    for source_path, variant in zip(source_paths, ablation_variants(args)):
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        current_dir = evaluate_current(args, source_path, variant)
        legacy_dir = run_legacy_wrapper(args, source_path, variant, simple=False)
        simple_dir = run_legacy_wrapper(args, source_path, variant, simple=True)
        print(f"{variant} current outputs: {current_dir}")
        print(f"{variant} legacy outputs: {legacy_dir}")
        print(f"{variant} simple outputs: {simple_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
