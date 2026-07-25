#!/usr/bin/env python3
"""Run the frozen c2bd847 pipeline with current model names.

This wrapper intentionally imports modules from ``legacy_src`` first, so the
prompt modules are the frozen copies from c2bd847 rather than current repo
prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable


BASELINE_DIR = Path(__file__).resolve().parent
LEGACY_SRC = BASELINE_DIR / "legacy_src"
RESULTS_DIR = BASELINE_DIR / "results"

DATASETS = ("massmaps", "cholec", "cardiac", "sepsis", "supernova", "emotion", "politeness")
METHODS = ("vanilla", "cot", "socratic", "subq")
CURRENT_MODELS = (
    "gpt-5.2-pro-2025-12-11",
    "gpt-5-mini-2025-08-07",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
)
DEFAULT_EVAL_MODEL = "gpt-5-mini-2025-08-07"


def configure_imports() -> None:
    sys.path.insert(0, str(LEGACY_SRC))
    os.environ.setdefault("CACHE_DIR", str(BASELINE_DIR / "llm_cache" / "diskcache"))
    os.environ.setdefault("LLMS_CACHE_PATH", str(BASELINE_DIR / "llm_cache" / "llms.py.cache"))


def clean_for_json(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return clean_for_json(value.to_dict())
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(v) for v in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__") and value.__class__.__module__ != "builtins":
        return clean_for_json(value.__dict__)
    return value


def write_results(dataset: str, method: str, model: str, eval_model: str, examples: Iterable[Any], output_dir: Path) -> Path:
    save_dir = output_dir / method
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    safe_eval = eval_model.replace("/", "_")
    save_path = save_dir / f"{dataset}_{safe_model}_eval-{safe_eval}.json"
    with save_path.open("w") as f:
        json.dump([clean_for_json(example) for example in examples], f, indent=4)
    return save_path


def choose_indices(length: int, num_samples: int, seed: int) -> list[int]:
    if num_samples <= 0 or num_samples >= length:
        return list(range(length))
    rng = random.Random(seed)
    return rng.sample(range(length), num_samples)


def run_cholec(args: argparse.Namespace) -> Path:
    import cholec

    dataset = cholec.CholecDataset(split="test", image_size=(360, 640))
    items = [dataset[i] for i in choose_indices(len(dataset), args.num_samples, args.seed)]
    examples = cholec.items_to_examples(
        items=items,
        explanation_model=args.model,
        evaluation_model=args.eval_model,
        baseline=args.method,
        verbose=args.verbose,
    )
    return write_results("cholec", args.method, args.model, args.eval_model, examples, args.output_dir)


def run_cardiac(args: argparse.Namespace) -> Path:
    import cardiac
    from datasets import load_dataset

    ds = load_dataset("BrachioLab/mcmed-cardiac")
    data = ds["train"].to_pandas()
    if args.num_samples > 0:
        data = data.sample(args.num_samples, random_state=args.seed).reset_index(drop=True)
    data["label"] = data["label"].map({True: "Yes", False: "No"})
    examples = cardiac.cardiac_data_to_examples(
        cardiac_data=data,
        explanation_model=args.model,
        evaluation_model=args.eval_model,
        baseline=args.method,
        verbose=args.verbose,
    )
    return write_results("cardiac", args.method, args.model, args.eval_model, examples, args.output_dir)


def run_emotion(args: argparse.Namespace) -> Path:
    import numpy as np
    import emotion

    data = emotion.load_emotion_data()
    if args.num_samples > 0:
        data = data.head(args.num_samples)
    examples = []
    for _, row in data.iterrows():
        label, explanation = emotion.get_llm_generated_answer(row["text"], args.method, args.model)
        if label is None:
            continue
        example = emotion.EmotionExample(
            text=row["text"],
            ground_truth=emotion.emotion_labels[int(row["labels"][0])],
            llm_label=label,
            llm_explanation=explanation,
        )
        example.accuracy = int(example.ground_truth == example.llm_label)
        claims = emotion.isolate_individual_features(example.llm_explanation, model=args.eval_model)
        example.claims = [claim.strip() for claim in claims or []]
        example.relevant_claims = emotion.distill_relevant_features(example, model=args.eval_model)
        for claim in example.relevant_claims:
            category, alignment_score, reasoning = emotion.calculate_expert_alignment_score(claim, model=args.eval_model)
            if category is None:
                continue
            example.alignment_scores.append(alignment_score)
            example.alignment_categories.append(category)
            example.alignment_reasonings.append(reasoning)
        example.final_alignment_score = float(np.mean(example.alignment_scores)) if example.alignment_scores else 0.0
        examples.append(example)
    return write_results("emotion", args.method, args.model, args.eval_model, examples, args.output_dir)


def run_politeness(args: argparse.Namespace) -> Path:
    import numpy as np
    import politeness

    data = politeness.load_politeness_data()
    if args.num_samples > 0:
        data = data.head(args.num_samples)
    examples = []
    for _, row in data.iterrows():
        rating, explanation = politeness.get_llm_generated_answer(row["Utterance"], args.method, args.model)
        if rating is None:
            continue
        example = politeness.PolitenessExample(
            utterance=row["Utterance"],
            ground_truth=float(row["politeness"]) + 3,
            llm_score=rating,
            llm_explanation=explanation,
        )
        example.mse = (example.ground_truth - example.llm_score) ** 2
        claims = politeness.isolate_individual_features(example.llm_explanation, model=args.eval_model)
        example.claims = [claim.strip() for claim in claims or []]
        example.relevant_claims = politeness.distill_relevant_features(example, model=args.eval_model)
        for claim in example.relevant_claims:
            category, alignment_score, reasoning = politeness.calculate_expert_alignment_score(claim, model=args.eval_model)
            if category is None:
                continue
            example.alignment_scores.append(alignment_score)
            example.alignment_categories.append(category)
            example.alignment_reasonings.append(reasoning)
        example.final_alignment_score = float(np.mean(example.alignment_scores)) if example.alignment_scores else 0.0
        examples.append(example)
    return write_results("politeness", args.method, args.model, args.eval_model, examples, args.output_dir)


def run_sepsis(args: argparse.Namespace) -> Path:
    import numpy as np
    import sepsis
    from datasets import load_dataset

    ds = load_dataset("BrachioLab/mcmed-sepsis")
    data = ds["test"].to_pandas()
    if args.num_samples > 0:
        data = data.sample(args.num_samples, random_state=args.seed).reset_index(drop=True)
    examples = []
    for _, row in data.iterrows():
        time_series = sepsis.parse_measurement_string(row["data"])
        label, explanation = sepsis.get_llm_generated_answer(time_series, args.method, model=args.model)
        if label is None:
            continue
        example = sepsis.SepsisExample(
            time_series_text=row["data"],
            time_series_data=time_series,
            ground_truth=row["label"],
            llm_label=label,
            llm_explanation=explanation,
        )
        claims = sepsis.isolate_individual_features(example.llm_explanation, model=args.eval_model)
        example.claims = [claim.strip() for claim in claims or []]
        example.relevant_claims = sepsis.distill_relevant_features(example, model=args.eval_model)
        for claim in example.relevant_claims:
            category, alignment_score, reasoning = sepsis.calculate_expert_alignment_score(claim, model=args.eval_model)
            if category is None:
                continue
            example.alignment_scores.append(alignment_score)
            example.alignment_categories.append(category)
            example.alignment_reasonings.append(reasoning)
        example.final_alignment_score = float(np.mean(example.alignment_scores)) if example.alignment_scores else 0.0
        examples.append(example)
    return write_results("sepsis", args.method, args.model, args.eval_model, examples, args.output_dir)


def run_supernova(args: argparse.Namespace) -> Path:
    import numpy as np
    import supernova
    from datasets import load_dataset

    ds = load_dataset("BrachioLab/supernova")
    data = ds["test"].to_pandas()
    if args.num_samples > 0:
        data = data.sample(args.num_samples, random_state=args.seed).reset_index(drop=True)
    examples = []
    for _, row in data.iterrows():
        label, explanation = supernova.get_llm_generated_answer(row["data"], args.method, model=args.model)
        if label is None:
            continue
        example = supernova.SupernovaExample(
            file=row["filename"],
            time_series_data=row["data"],
            ground_truth=row["label"],
            llm_label=label,
            llm_explanation=explanation,
        )
        claims = supernova.isolate_individual_features(example.llm_explanation, model=args.eval_model)
        example.claims = [claim.strip() for claim in claims or []]
        example.relevant_claims = supernova.distill_relevant_features(
            example.time_series_data,
            example.llm_label,
            example.claims,
            model=args.eval_model,
            verbose=args.verbose,
        )
        for claim in example.relevant_claims:
            category, alignment_score, reasoning = supernova.calculate_expert_alignment_score(claim, model=args.eval_model)
            if category is None:
                continue
            example.alignment_scores.append(alignment_score)
            example.alignment_categories.append(category)
            example.alignment_reasonings.append(reasoning)
        example.final_alignment_score = float(np.mean(example.alignment_scores)) if example.alignment_scores else 0.0
        examples.append(example)
    return write_results("supernova", args.method, args.model, args.eval_model, examples, args.output_dir)


def run_massmaps(args: argparse.Namespace) -> Path:
    import numpy as np
    import massmaps
    from datasets import load_dataset

    ds = load_dataset("BrachioLab/massmaps-cosmogrid-100k", split="test")
    ds.set_format("torch", columns=["input", "label"])
    examples = []
    for idx in choose_indices(len(ds), args.num_samples, args.seed):
        row = ds[idx]
        label = row["label"]
        answer = {"Omega_m": float(label[0]), "sigma_8": float(label[1])}
        llm_answer, explanation = massmaps.get_llm_generated_answer(row["input"], args.method, model=args.model)
        if llm_answer is None:
            continue
        example = massmaps.MassMapsExample(row["input"], answer, llm_answer, explanation)
        claims = massmaps.isolate_individual_features(example.llm_explanation, model=args.eval_model)
        example.claims = [claim.strip() for claim in claims or []]
        example.relevant_claims = massmaps.distill_relevant_features(
            example.input,
            example.llm_answer,
            example.claims,
            model=args.eval_model,
            verbose=args.verbose,
        )
        align_infos = massmaps.calculate_expert_alignment_scores(example.relevant_claims, model=args.eval_model)
        example.alignment_categories = [info["Category"] for info in align_infos]
        example.alignment_scores = [info["Alignment"] for info in align_infos]
        example.alignment_reasonings = [info["Reasoning"] for info in align_infos]
        example.final_alignment_score = float(np.mean(example.alignment_scores)) if example.alignment_scores else 0.0
        examples.append(example)
    return write_results("massmaps", args.method, args.model, args.eval_model, examples, args.output_dir)


RUNNERS = {
    "massmaps": run_massmaps,
    "cholec": run_cholec,
    "cardiac": run_cardiac,
    "sepsis": run_sepsis,
    "supernova": run_supernova,
    "emotion": run_emotion,
    "politeness": run_politeness,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen c2bd847 previous pipeline.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", "--baseline", dest="method", choices=METHODS, default="vanilla")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--num-samples", type=int, default=3, help="0 means all available examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate dispatch/import path without loading data or calling APIs.")
    parser.add_argument("--list-models", action="store_true", help="Print current default model names and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_models:
        print("\n".join(CURRENT_MODELS))
        return 0

    configure_imports()
    args.output_dir = args.output_dir.resolve()

    if args.dry_run:
        print(f"baseline_dir={BASELINE_DIR}")
        print(f"legacy_src={LEGACY_SRC}")
        print(f"dataset={args.dataset}")
        print(f"method={args.method}")
        print(f"model={args.model}")
        print(f"eval_model={args.eval_model}")
        return 0

    save_path = RUNNERS[args.dataset](args)
    print(f"saved {args.dataset} results to {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
