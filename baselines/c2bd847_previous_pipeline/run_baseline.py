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

MASSMAPS_EXPERT_CRITERIA = """1. Lensing Peak (Cluster) Abundance: A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8, since a clumpier matter distribution produces more frequent massive halos.
2. Void Size and Frequency: Extensive low-convergence void regions suggest a lower Omega_m, as a reduced overall matter density allows bigger underdense expanses to form in the cosmic web.
3. Filament Thickness and Sharpness: Bold, sharply defined filaments threading between clusters imply a higher sigma_8 (stronger small-scale clustering), whereas thin or diffuse filaments point to a lower amplitude of matter fluctuations.
4. Fine-Scale Clumpiness: A grainy, fine-textured pattern of small-scale lensing fluctuations (many mini-clumps) is a visual signature of high sigma_8, whereas a smoother, more homogeneous map suggests a lower sigma_8.
5. Connectivity of the Cosmic Web: A highly interconnected filament network (with filaments linking most clusters into a continuous web) hints at a higher Omega_m, whereas a more fragmented scene of isolated clumps separated by wide gaps is expected for a lower Omega_m.
6. Density Contrast Extremes: Very pronounced contrast between dense regions and empty voids - i.e. bright lensing peaks adjacent to dark void areas - signals an enhanced variance of the density field (high sigma_8), whereas subdued contrast suggests lower sigma_8."""

MASSMAPS_SIMPLE_ALIGNMENT_PROMPT = """You will be given an explanation for why predictions for Omega_m and sigma_8 values were given to a weak lensing mass map. You will also be given expert cosmology criteria that should be used to judge this type of explanation.

Your task is as follows:
1. Reason about how well the explanation uses the expert criteria.
2. Give a single alignment score from 0 to 1.

Scoring guidance:
1.0 means the explanation is specific, directly relevant, and strongly grounded in the expert criteria.
0.5 means the explanation uses some relevant expert criteria but is incomplete, vague, noisy, or only partially connected to the prediction.
0.0 means the explanation is unrelated to the expert criteria, mostly unsupported, or draws conclusions opposite to the criteria.

Return your answer exactly as:
Reasoning: <A brief explanation of the score.>
Score: <a number between 0 and 1>

-----
Expert cosmology criteria:
{criteria}
-----

Prediction:
Omega_m = {omega_m}, sigma_8 = {sigma_8}

Explanation:
{explanation}
"""

CHOLEC_EXPERT_CRITERIA = """1. Calot's triangle cleared: Hepatocystic triangle must be fully cleared of fat/fibrosis so that its boundaries are unmistakable.
2. Cystic plate exposed: The lower third of the gallbladder must be dissected off the liver to reveal the shiny cystic plate and ensure the correct dissection plane.
3. Only two structures visible: Only the cystic duct and cystic artery should be seen entering the gallbladder before any clipping or cutting.
4. Above the R4U line: Dissection must remain cephalad to an imaginary line from Rouviere's sulcus to liver segment IV umbilical fissure to avoid the common bile duct.
5. Infundibulum start point: Dissection can begin at the gallbladder infundibulum-cystic duct junction, at the lateral or medial gallbladder edges above Rouviere's sulcus, or along the cystic plate.
6. Peritoneal plane: When separating the gallbladder from the liver, stay in the avascular peritoneal cleavage plane.
7. Cystic lymph node guide: Identify the cystic lymph node and clip the artery on the gallbladder side of the node to avoid injuring the hepatic artery.
8. No division without ID: Never divide any duct or vessel until it is unequivocally identified as the cystic structure entering the gallbladder.
9. Inflammation bailout: If dense scarring or distorted anatomy obscures Calot's triangle, convert to open surgery or a fenestrated subtotal approach rather than blind cutting."""

CHOLEC_SIMPLE_ALIGNMENT_PROMPT = """You will be given an explanation for why safe and unsafe regions were predicted in a laparoscopic cholecystectomy image. You will also be given expert surgical safety criteria that should be used to judge this type of explanation.

Your task is as follows:
1. Reason about how well the explanation uses the expert criteria.
2. Give a single alignment score from 0 to 1.

Scoring guidance:
1.0 means the explanation is specific, directly relevant, and strongly grounded in the expert criteria.
0.5 means the explanation uses some relevant expert criteria but is incomplete, vague, noisy, or only partially connected to the prediction.
0.0 means the explanation is unrelated to the expert criteria, mostly unsupported, or draws conclusions opposite to the criteria.

Return your answer exactly as:
Reasoning: <A brief explanation of the score.>
Score: <a number between 0 and 1>

-----
Expert surgical safety criteria:
{criteria}
-----

Predicted safe grid cells:
{safe_list}

Predicted unsafe grid cells:
{unsafe_list}

Explanation:
{explanation}
"""


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


def evaluate_existing_massmaps_explanations(args: argparse.Namespace) -> Path:
    import json

    import numpy as np
    import torch
    from tqdm.auto import tqdm

    import massmaps

    input_path = args.input_path or args.input_dir
    if input_path is None:
        input_path = BASELINE_DIR.parent.parent / "results" / args.method / f"massmaps_{args.model.replace('/', '_')}.json"
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Massmaps explanation result file does not exist: {input_path}")

    with input_path.open("rt") as input_file:
        all_results = json.load(input_file)
    if not isinstance(all_results, list):
        raise ValueError(f"Expected a JSON list in {input_path}")
    if args.num_samples > 0:
        all_results = all_results[: args.num_samples]

    eval_model_name = args.eval_model.replace("/", "_")
    explanation_model_name = args.model.replace("/", "_")

    output_dir = args.output_dir
    if output_dir == RESULTS_DIR:
        output_dir = (
            BASELINE_DIR
            / "notebooks"
            / "_dump"
            / "massmaps"
            / "final"
            / "legacy-c2bd847"
            / explanation_model_name
            / args.method
            / f"eval.{eval_model_name}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for fallback_idx, example_dict in enumerate(tqdm(all_results)):
        example_idx = example_dict.get("idx", fallback_idx)
        filename = f"{example_idx}.json"
        save_path = output_dir / filename
        if save_path.exists():
            print(f"exists, skipping: {save_path}")
            continue

        if not isinstance(example_dict["input"], torch.Tensor):
            example_dict["input"] = torch.tensor(example_dict["input"])

        example = massmaps.MassMapsExample(
            input=example_dict["input"],
            answer=example_dict["answer"],
            llm_answer=example_dict["llm_answer"],
            llm_explanation=example_dict["llm_explanation"],
        )
        example.__dict__ = example_dict
        example.idx = example_idx

        claims = massmaps.isolate_individual_features(example.llm_explanation, model=args.eval_model)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]

        example.relevant_claims = massmaps.distill_relevant_features(
            example.input,
            example.llm_answer,
            example.claims,
            model=args.eval_model,
            verbose=args.verbose,
        )

        align_infos = massmaps.calculate_expert_alignment_scores(
            example.relevant_claims,
            model=args.eval_model,
        )
        example.alignment_categories = [info["Category"] for info in align_infos]
        example.alignment_scores = [info["Alignment"] for info in align_infos]
        example.alignment_reasonings = [info["Reasoning"] for info in align_infos]
        final_alignment_score = np.mean(example.alignment_scores) if example.alignment_scores else np.nan
        example.final_alignment_score = None if np.isnan(final_alignment_score) else float(final_alignment_score)

        save_dict = {}
        for key, value in example.__dict__.items():
            save_dict[key] = value if not isinstance(value, torch.Tensor) else value.cpu().numpy().tolist()
        with save_path.open("wt") as output_file:
            json.dump(clean_for_json(save_dict), output_file, indent=4)

    return output_dir


def get_cholec_item_lookup():
    import cholec

    dataset = cholec.CholecDataset(split="test", image_size=(360, 640))
    return {dataset[i]["id"]: dataset[i] for i in range(len(dataset))}


def evaluate_existing_cholec_explanations(args: argparse.Namespace) -> Path:
    import json

    from tqdm.auto import tqdm

    import cholec

    input_path = args.input_path or args.input_dir
    if input_path is None:
        input_path = BASELINE_DIR.parent.parent / "results" / args.method / f"cholec_{args.model.replace('/', '_')}.json"
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Cholec explanation result file does not exist: {input_path}")

    with input_path.open("rt") as input_file:
        all_results = json.load(input_file)
    if not isinstance(all_results, list):
        raise ValueError(f"Expected a JSON list in {input_path}")
    if args.num_samples > 0:
        all_results = all_results[: args.num_samples]

    eval_model_name = args.eval_model.replace("/", "_")
    explanation_model_name = args.model.replace("/", "_")

    output_dir = args.output_dir
    if output_dir == RESULTS_DIR:
        output_dir = (
            BASELINE_DIR
            / "notebooks"
            / "_dump"
            / "cholec"
            / "final"
            / "legacy-c2bd847"
            / explanation_model_name
            / args.method
            / f"eval.{eval_model_name}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    item_by_id = get_cholec_item_lookup()
    for fallback_idx, example_dict in enumerate(tqdm(all_results)):
        example_idx = example_dict.get("idx", fallback_idx)
        filename = example_dict.get("_filename") or f"{example_idx}.json"
        save_path = output_dir / Path(filename).name
        if save_path.exists():
            print(f"exists, skipping: {save_path}")
            continue

        item = item_by_id.get(example_dict.get("id"))
        if item is None:
            raise KeyError(f"Could not find Cholec dataset item with id={example_dict.get('id')!r}")

        example = cholec.CholecExample(
            id=example_dict["id"],
            image=item["image"],
            true_safe_list=example_dict.get("true_safe_list", []),
            true_unsafe_list=example_dict.get("true_unsafe_list", []),
            llm_raw_output=example_dict.get("llm_raw_output", ""),
            llm_explanation=example_dict.get("llm_explanation", ""),
            llm_safe_list=example_dict.get("llm_safe_list", []),
            llm_unsafe_list=example_dict.get("llm_unsafe_list", []),
        )
        example.__dict__.update(example_dict)
        example.image = item["image"]

        claims = cholec.isolate_individual_features(example.llm_explanation, model=args.eval_model)
        example.all_claims = [claim.strip() for claim in claims or []]
        example.relevant_claims = cholec.distill_relevant_features(
            example.image,
            example.all_claims,
            model=args.eval_model,
        )

        align_infos = cholec.calculate_expert_alignment_scores(
            example.relevant_claims,
            model=args.eval_model,
        )
        example.alignable_claims = [info["Claim"] for info in align_infos]
        example.aligned_category_ids = [info["Category ID"] for info in align_infos]
        example.alignment_scores = [info["Alignment"] for info in align_infos]
        example.alignment_reasonings = [info["Reasoning"] for info in align_infos]
        example.final_alignment_score = (
            sum(info["Alignment"] for info in align_infos) / len(example.all_claims)
            if example.all_claims
            else 0.0
        )
        example.eval_model = args.eval_model
        example.explanation_model = args.model
        example.generation_method = args.method
        example.legacy_evaluation_mode = "legacy"

        save_dict = dict(example.__dict__)
        save_dict.pop("image", None)
        with save_path.open("wt") as output_file:
            json.dump(clean_for_json(save_dict), output_file, indent=4)

    return output_dir


def parse_simple_score(response: str) -> tuple[str, float]:
    import re

    text = "" if response is None else str(response)
    reasoning = text.strip()
    score = None

    score_match = re.search(r"(?im)^\s*score\s*[:\-]\s*([01](?:\.\d+)?)\s*$", text)
    if score_match:
        score = float(score_match.group(1))
    else:
        numbers = [float(match) for match in re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text)]
        if numbers:
            score = numbers[-1]

    if score is None:
        score = 0.0
    score = max(0.0, min(1.0, score))

    reasoning_match = re.search(r"(?ims)^\s*reasoning\s*[:\-]\s*(.*?)(?:^\s*score\s*[:\-]|\Z)", text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return reasoning, score


def evaluate_existing_massmaps_simple(args: argparse.Namespace) -> Path:
    import json

    from tqdm.auto import tqdm

    import massmaps

    input_path = args.input_path or args.input_dir
    if input_path is None:
        input_path = BASELINE_DIR.parent.parent / "results" / args.method / f"massmaps_{args.model.replace('/', '_')}.json"
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Massmaps explanation result file does not exist: {input_path}")

    with input_path.open("rt") as input_file:
        all_results = json.load(input_file)
    if not isinstance(all_results, list):
        raise ValueError(f"Expected a JSON list in {input_path}")
    if args.num_samples > 0:
        all_results = all_results[: args.num_samples]

    eval_model_name = args.eval_model.replace("/", "_")
    explanation_model_name = args.model.replace("/", "_")

    output_dir = args.output_dir
    if output_dir == RESULTS_DIR:
        output_dir = (
            BASELINE_DIR
            / "notebooks"
            / "_dump"
            / "massmaps"
            / "final"
            / "legacy-c2bd847-simple"
            / explanation_model_name
            / args.method
            / f"eval.simple.{eval_model_name}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = massmaps.load_model(args.eval_model)
    for fallback_idx, example_dict in enumerate(tqdm(all_results)):
        example_idx = example_dict.get("idx", fallback_idx)
        save_path = output_dir / f"{example_idx}.json"
        if save_path.exists():
            print(f"exists, skipping: {save_path}")
            continue

        llm_answer = example_dict.get("llm_answer") or {}
        prompt = MASSMAPS_SIMPLE_ALIGNMENT_PROMPT.format(
            criteria=MASSMAPS_EXPERT_CRITERIA,
            omega_m=llm_answer.get("Omega_m", "N/A"),
            sigma_8=llm_answer.get("sigma_8", "N/A"),
            explanation=example_dict.get("llm_explanation", ""),
        )
        raw_output = llm(prompt)
        reasoning, score = parse_simple_score(raw_output)

        save_dict = dict(example_dict)
        save_dict["legacy_evaluation_mode"] = "simple"
        save_dict["simple_alignment_raw_output"] = raw_output
        save_dict["simple_alignment_reasoning"] = reasoning
        save_dict["simple_alignment_score"] = score
        save_dict["final_alignment_score"] = score
        save_dict["eval_model"] = args.eval_model
        save_dict["explanation_model"] = args.model
        save_dict["generation_method"] = args.method

        with save_path.open("wt") as output_file:
            json.dump(clean_for_json(save_dict), output_file, indent=4)

    return output_dir


def evaluate_existing_cholec_simple(args: argparse.Namespace) -> Path:
    import json

    from tqdm.auto import tqdm

    import cholec

    input_path = args.input_path or args.input_dir
    if input_path is None:
        input_path = BASELINE_DIR.parent.parent / "results" / args.method / f"cholec_{args.model.replace('/', '_')}.json"
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Cholec explanation result file does not exist: {input_path}")

    with input_path.open("rt") as input_file:
        all_results = json.load(input_file)
    if not isinstance(all_results, list):
        raise ValueError(f"Expected a JSON list in {input_path}")
    if args.num_samples > 0:
        all_results = all_results[: args.num_samples]

    eval_model_name = args.eval_model.replace("/", "_")
    explanation_model_name = args.model.replace("/", "_")

    output_dir = args.output_dir
    if output_dir == RESULTS_DIR:
        output_dir = (
            BASELINE_DIR
            / "notebooks"
            / "_dump"
            / "cholec"
            / "final"
            / "legacy-c2bd847-simple"
            / explanation_model_name
            / args.method
            / f"eval.simple.{eval_model_name}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = cholec.load_model(args.eval_model)
    for fallback_idx, example_dict in enumerate(tqdm(all_results)):
        example_idx = example_dict.get("idx", fallback_idx)
        filename = example_dict.get("_filename") or f"{example_idx}.json"
        save_path = output_dir / Path(filename).name
        if save_path.exists():
            print(f"exists, skipping: {save_path}")
            continue

        prompt = CHOLEC_SIMPLE_ALIGNMENT_PROMPT.format(
            criteria=CHOLEC_EXPERT_CRITERIA,
            safe_list=example_dict.get("llm_safe_list", []),
            unsafe_list=example_dict.get("llm_unsafe_list", []),
            explanation=example_dict.get("llm_explanation", ""),
        )
        raw_output = llm(prompt)
        reasoning, score = parse_simple_score(raw_output)

        save_dict = dict(example_dict)
        save_dict["legacy_evaluation_mode"] = "simple"
        save_dict["simple_alignment_raw_output"] = raw_output
        save_dict["simple_alignment_reasoning"] = reasoning
        save_dict["simple_alignment_score"] = score
        save_dict["final_alignment_score"] = score
        save_dict["eval_model"] = args.eval_model
        save_dict["explanation_model"] = args.model
        save_dict["generation_method"] = args.method

        with save_path.open("wt") as output_file:
            json.dump(clean_for_json(save_dict), output_file, indent=4)

    return output_dir


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
    parser.add_argument("--input-path", type=Path, help="Existing aggregate result JSON to reuse for evaluation-only runs.")
    parser.add_argument("--input-dir", type=Path, help="Deprecated alias for --input-path.")
    parser.add_argument("--evaluate-existing-explanations", action="store_true", help="Evaluate existing aggregate results without rerunning generation.")
    parser.add_argument("--simple-baseline", action="store_true", help="Score each existing explanation with a simple criteria prompt.")
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
        if args.input_path or args.input_dir:
            print(f"input_path={args.input_path or args.input_dir}")
        print(f"evaluate_existing_explanations={args.evaluate_existing_explanations}")
        print(f"simple_baseline={args.simple_baseline}")
        return 0

    if args.simple_baseline:
        if args.dataset == "massmaps":
            save_path = evaluate_existing_massmaps_simple(args)
        elif args.dataset == "cholec":
            save_path = evaluate_existing_cholec_simple(args)
        else:
            raise ValueError("--simple-baseline is currently implemented only for massmaps and cholec")
        print(f"saved legacy simple {args.dataset} evaluation results under {save_path}")
        return 0

    if args.evaluate_existing_explanations:
        if args.dataset == "massmaps":
            save_path = evaluate_existing_massmaps_explanations(args)
        elif args.dataset == "cholec":
            save_path = evaluate_existing_cholec_explanations(args)
        else:
            raise ValueError("--evaluate-existing-explanations is currently implemented only for massmaps and cholec")
        print(f"saved legacy {args.dataset} evaluation results under {save_path}")
        return 0

    save_path = RUNNERS[args.dataset](args)
    print(f"saved {args.dataset} results to {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
