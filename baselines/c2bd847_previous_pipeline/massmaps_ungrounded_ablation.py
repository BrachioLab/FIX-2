#!/usr/bin/env python3
"""Run massmaps ungrounded-criteria claim ablations.

The ablation appends a generated claim that aligns with one expert criterion
but is not grounded in the input image. Source result JSONs are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from massmaps_duplication_ablation import (
    evaluate_current,
    run_legacy_wrapper,
    safe_model_name,
    default_source_path,
    load_json,
    write_json,
)


BASELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASELINE_DIR.parents[1]
SRC_DIR = ROOT_DIR / "src"

MASSMAPS_EXPERT_CRITERIA = """1. Lensing Peak (Cluster) Abundance: A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8, since a clumpier matter distribution produces more frequent massive halos.
2. Void Size and Frequency: Extensive low-convergence void regions suggest a lower Omega_m, as a reduced overall matter density allows bigger underdense expanses to form in the cosmic web.
3. Filament Thickness and Sharpness: Bold, sharply defined filaments threading between clusters imply a higher sigma_8 (stronger small-scale clustering), whereas thin or diffuse filaments point to a lower amplitude of matter fluctuations.
4. Fine-Scale Clumpiness: A grainy, fine-textured pattern of small-scale lensing fluctuations (many mini-clumps) is a visual signature of high sigma_8, whereas a smoother, more homogeneous map suggests a lower sigma_8.
5. Connectivity of the Cosmic Web: A highly interconnected filament network (with filaments linking most clusters into a continuous web) hints at a higher Omega_m, whereas a more fragmented scene of isolated clumps separated by wide gaps is expected for a lower Omega_m.
6. Density Contrast Extremes: Very pronounced contrast between dense regions and empty voids - i.e. bright lensing peaks adjacent to dark void areas - signals an enhanced variance of the density field (high sigma_8), whereas subdued contrast suggests lower sigma_8."""

UNGROUNDED_ABLATION_PROMPT = """You will be given:
- A weak lensing mass map image.
- The original explanation text for predicted Omega_m and sigma_8 values.
- Atomic claims extracted from that explanation.
- Filtered claims judged relevant to the prediction.
- Expert cosmology criteria.

Your task is to write one additional atomic claim for an ablation.

Requirements:
- The claim must align with exactly one of the expert cosmology criteria.
- The claim must be ungrounded in the input image: it should introduce a visual observation that is not supported by the image, original explanation, atomic claims, or filtered claims.
- The claim should sound plausible and should be standalone.
- Do not contradict the requested output format.
- Do not explain that the claim is ungrounded.
- Output exactly two lines in the format shown below.

Here are some examples:

Example 1:
INPUT SUMMARY: The explanation says the map contains mostly gray and blue regions with only a few small red spots. The claims do not mention yellow peaks.
TARGET CRITERIA:
1. Lensing Peak (Cluster) Abundance: A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8.

OUTPUT:
CRITERION: Lensing Peak (Cluster) Abundance
CLAIM: The map contains numerous bright yellow high-convergence peaks, indicating a high abundance of massive lensing clusters.

Example 2:
INPUT SUMMARY: The explanation discusses several compact peaks and moderate contrast. The claims do not mention an interconnected filament network.
TARGET CRITERIA:
5. Connectivity of the Cosmic Web: A highly interconnected filament network hints at a higher Omega_m.

OUTPUT:
CRITERION: Connectivity of the Cosmic Web
CLAIM: The map shows a continuous web of filaments linking most dense regions, suggesting elevated Omega_m.

Now write one ungrounded claim for the following example.

ORIGINAL EXPLANATION:
{explanation}

ATOMIC CLAIMS:
{claims}

FILTERED CLAIMS:
{filtered_claims}

EXPERT COSMOLOGY CRITERIA:
{criteria}

OUTPUT:
"""


def load_llm(model: str):
    sys.path.insert(0, str(SRC_DIR))
    from llms import load_model

    return load_model(model)


def parse_ungrounded_claim(raw_output: str) -> tuple[str, str]:
    criterion = ""
    claim = ""
    criterion_match = re.search(r"(?im)^\s*CRITERION\s*:\s*(.+?)\s*$", raw_output or "")
    claim_match = re.search(r"(?im)^\s*CLAIM\s*:\s*(.+?)\s*$", raw_output or "")
    if criterion_match:
        criterion = criterion_match.group(1).strip()
    if claim_match:
        claim = claim_match.group(1).strip()
    if not claim:
        lines = [line.strip() for line in (raw_output or "").splitlines() if line.strip()]
        claim = re.sub(r"(?i)^claim\s*:\s*", "", lines[-1]).strip() if lines else ""
    return criterion, claim


def generate_ungrounded_sources(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(SRC_DIR))
    import massmaps

    source_path = (args.source_path or default_source_path(args.method, args.model, args.eval_model)).resolve()
    source_rows = load_json(source_path)
    if not isinstance(source_rows, list):
        raise ValueError(f"Expected JSON list in {source_path}")
    if args.num_samples > 0:
        source_rows = source_rows[: args.num_samples]

    output_path = args.ablation_dir / "ungrounded_criteria" / f"massmaps_{safe_model_name(args.model)}_{args.method}_ungrounded.json"
    output_path = output_path.resolve()
    generated_dir = (
        args.generated_dir
        / "ungrounded_criteria"
        / safe_model_name(args.model)
        / args.method
        / f"generator.{safe_model_name(args.generation_model)}"
    ).resolve()

    llm = load_llm(args.generation_model)
    ablated_rows = []
    for fallback_idx, row in enumerate(source_rows):
        row_copy = deepcopy(row)
        idx = int(row_copy.get("idx", fallback_idx))
        record_path = generated_dir / f"{idx}.json"

        if record_path.exists() and not args.overwrite:
            generation_record = load_json(record_path)
        else:
            image_tensor = torch.tensor(row_copy["input"])
            image = massmaps.massmap_to_pil_norm(image_tensor)
            prompt = UNGROUNDED_ABLATION_PROMPT.format(
                explanation=row_copy.get("llm_explanation", ""),
                claims="\n".join(row_copy.get("claims") or []),
                filtered_claims="\n".join(row_copy.get("relevant_claims") or []),
                criteria=MASSMAPS_EXPERT_CRITERIA,
            )
            raw_output = llm((prompt, image))
            criterion, ungrounded_claim = parse_ungrounded_claim(raw_output)
            if not ungrounded_claim:
                raise ValueError(f"Could not parse ungrounded claim for idx={idx}: {raw_output}")

            original_explanation = row_copy.get("llm_explanation", "")
            ablated_explanation = f"{original_explanation.rstrip()}\n\n{ungrounded_claim}"
            generation_record = {
                "idx": idx,
                "variant": "ungrounded_criteria",
                "source_path": str(source_path),
                "explanation_model": args.model,
                "evaluation_model": args.eval_model,
                "generation_method": args.method,
                "generation_model": args.generation_model,
                "criteria": MASSMAPS_EXPERT_CRITERIA,
                "original_explanation": original_explanation,
                "atomic_claims": row_copy.get("claims") or [],
                "filtered_claims": row_copy.get("relevant_claims") or [],
                "prompt": prompt,
                "raw_output": raw_output,
                "selected_criterion": criterion,
                "ungrounded_claim": ungrounded_claim,
                "ablated_explanation": ablated_explanation,
            }
            write_json(record_path, generation_record)

        row_copy["llm_explanation_original"] = generation_record["original_explanation"]
        row_copy["llm_explanation"] = generation_record["ablated_explanation"]
        row_copy["ungrounded_ablation"] = {
            "variant": "ungrounded_criteria",
            "generation_model": generation_record["generation_model"],
            "selected_criterion": generation_record["selected_criterion"],
            "raw_output": generation_record["raw_output"],
            "ungrounded_claim": generation_record["ungrounded_claim"],
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
    print(f"saved ungrounded ablation source: {output_path}")
    print(f"saved ungrounded generation records under: {generated_dir}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run massmaps ungrounded-criteria ablation.")
    parser.add_argument("--method", default="vanilla")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07", help="Explanation model.")
    parser.add_argument("--eval-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--generation-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--ablation-dir", type=Path, default=BASELINE_DIR / "results" / "ungrounded_ablation" / "sources")
    parser.add_argument("--generated-dir", type=Path, default=BASELINE_DIR / "results" / "ungrounded_ablation" / "generated")
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR / "results" / "ungrounded_ablation" / "evaluations")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.ablation_dir = args.ablation_dir.resolve()
    args.generated_dir = args.generated_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.evaluate_only:
        source_path = (
            args.ablation_dir
            / "ungrounded_criteria"
            / f"massmaps_{safe_model_name(args.model)}_{args.method}_ungrounded.json"
        ).resolve()
    else:
        source_path = generate_ungrounded_sources(args)

    if args.generate_only:
        return 0

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    variant = "ungrounded_criteria"
    current_dir = evaluate_current(args, source_path, variant)
    legacy_dir = run_legacy_wrapper(args, source_path, variant, simple=False)
    simple_dir = run_legacy_wrapper(args, source_path, variant, simple=True)
    print(f"{variant} current outputs: {current_dir}")
    print(f"{variant} legacy outputs: {legacy_dir}")
    print(f"{variant} simple outputs: {simple_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
