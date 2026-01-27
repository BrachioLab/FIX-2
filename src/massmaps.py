import PIL
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import torch

import openai

import re
import json

from diskcache import Cache
from typing import Tuple
import random
import time
import math
from tqdm.auto import tqdm
from matplotlib.colors import LinearSegmentedColormap

from pathlib import Path
from PIL import Image
import PIL
from llms import load_model

from typing import Callable, Any


import argparse
import glob
import os

import json
import os
from collections import defaultdict
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error

from prompts.explanations import massmaps_prompt, vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline
from prompts.claim_decomposition import decomposition_massmaps
from prompts.relevance_filtering import relevance_massmaps, load_relevance_massmaps_prompt
from prompts.expert_alignment import alignment_massmaps
from prompts.category_mapping import category_mapping_massmaps
from prompts.claim_grouping import claim_grouping_massmaps
from prompts.expert_category_alignment import category_alignment_massmaps

cache = Cache(os.environ.get("CACHE_DIR"))

categories_list = [name for name, _ in sorted(category_mapping_massmaps["name2id"].items(), key=lambda x: x[1])]

class MassMapsExample:
    def __init__(self, input, answer, llm_answer, llm_explanation):
        self.input = input
        self.answer = answer
        self.llm_answer = llm_answer # this is the llm answer
        self.llm_explanation = llm_explanation
        self.claims = []
        self.relevant_claims = []
        self.alignment_scores = []
        self.alignment_categories = []
        self.alignment_reasonings = []

def convert_pil_to_base64(pil_image):
    """
    Converts a PIL image to a base64-encoded string.
    """
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    pil_image.load()

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_custom_colormap(colors=None):
    if colors is None:
        colors = [
            (-3, "blue"),
            (0,   "gray"),
            (2.9, "red"),
            (3,   "yellow"),
            (20,  "white"),
        ]
    positions, color_vals = zip(*colors)
    minp, maxp = min(positions), max(positions)
    positions = [(p-minp)/(maxp-minp) for p in positions]
    return LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, color_vals)))


def massmap_to_pil_norm(
    tensor: torch.Tensor,
    mean_center: bool = False,
    vmin: float = -3,
    vmax: float = 20,
    colors: list = None,
    scale: int = 11          # ← new: integer zoom factor
) -> Image.Image:
    """
    Convert a (1,H,W) tensor → PIL Image (H×W), with:
      • optional mean-centering
      • divide-by-std normalization
      • clip to [vmin,vmax], then min–max to [0,1]
      • apply custom colormap ⇒ RGB
      • optional nearest-neighbor up-scaling by integer *scale*
    """
    # 1) pull out H×W array
    arr = tensor.detach().cpu().numpy()[0]  # shape (H, W)

    # 2) normalize
    if mean_center:
        arr = arr - arr.mean()
    arr = arr / (arr.std() + 1e-8)

    # 3) clip & rescale to [0,1]
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin)

    # 4) colormap
    cmap = get_custom_colormap(colors)
    rgba = cmap(arr)                # (H, W, 4) floats in [0,1]
    rgb  = (rgba[..., :3] * 255).astype(np.uint8)

    # 5) make PIL Image
    img = Image.fromarray(rgb)

    # 6) optional nearest-neighbor enlargement
    if scale > 1:
        new_size = (img.width * scale, img.height * scale)
        img = img.resize(new_size, resample=Image.NEAREST)

    return img

@cache.memoize()
def get_llm_output(prompt, images=None, model='gpt-4o'):
    """
    prompt: str
    images: list of PIL images
    system_prompt: str
    """

    llm = load_model(model)

    result = llm([(prompt, *images)])[0]
    # print('result: ', result)
    # import pdb; pdb.set_trace()
    return result

_number_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_float(s: str) -> float:
    """
    Extract the first numeric token from *s* and return it as a float.
    Examples
    --------
    >>> parse_float("0.8.")      # → 0.8
    >>> parse_float("  1.23e-4") # → 1.23e-4
    """
    m = _number_pat.search(s)
    if not m:
        raise ValueError(f"No numeric value found in {s!r}")
    return float(m.group())

# def get_llm_generated_answer(
#     example: list[str] | str | torch.Tensor, #Image | Timeseries,
#     method: str = "vanilla",
#     model: str = "gpt-4o",
#     massmap_to_pil_norm: Callable = massmap_to_pil_norm,
# ) -> str:
#     """
#     Args:
#         example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
#           e.g., the emotion classification task.
#     """

#     if method == 'least_to_most':
#         method = 'subq'

#     if method == "vanilla":
#         prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", '')
#     elif method == "cot":
#         prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", cot_baseline)
#     elif method == "socratic":
#         prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", socratic_baseline)
#     elif method == "subq":
#         prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline)
#     else:
#         raise ValueError(f"Invalid method: {method}")

#     prompt = prompt.replace(
#         '[LAST_IMAGE_NUM]',
#         '1'
#     )
    

#     image_pil = [massmap_to_pil_norm(example)]

#     llm_response = get_llm_output(prompt, image_pil, model=model)

#     response_split = [r.strip() for r in llm_response.split("\n") if r.strip() != "" \
#         and r.strip().startswith("Explanation:") or r.strip().startswith("Prediction:")]
#     try:
        
#         explanation = response_split[0].split("Explanation: ")[1].strip()
#         answer = response_split[-1].split("Prediction: ")[1].strip()
#         # split the answer into Omega_m and sigma_8
#         answer = answer.split(", ")
#         answer = {
#             answer[0].split(": ")[0]: parse_float(answer[0].split(": ")[1]), 
#             answer[1].split(": ")[0]: parse_float(answer[1].split(": ")[1])
#         }
        
#         return answer, explanation
#     except Exception as e:
#         print("exception: ", e)
#         print(f"Error in parsing response {llm_response}")
#         import pdb; pdb.set_trace()
#         return None, None

def get_llm_generated_answer(
    image: torch.Tensor | np.ndarray | PIL.Image.Image | list[Any],
    method: str = "vanilla",
    model: str = "gpt-4o",
    massmap_to_pil_norm: Callable = massmap_to_pil_norm,
) -> str:
    if method == 'least_to_most':
        method = 'subq'

    if method == "vanilla":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", vanilla_baseline)
    elif method == "cot":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", cot_baseline)
    elif method == "socratic":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", socratic_baseline)
    elif method == "subq":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid method: {method}")

    llm = load_model(model)

    def process_response(response: str):
        response_split = [
            r.strip()
            for r in response.split("\n")
            if r.strip() != ""
            and (r.strip().startswith("Explanation:") or r.strip().startswith("Prediction:"))
        ]
        
        try:
            if (
                len(response_split) < 2
                or not response_split[0].startswith("Explanation:")
                or not response_split[-1].startswith("Prediction:")
            ):
                explanation = response.split("Prediction:")[0].split("Explanation:")[-1].strip()
                answer_text = response.split("Prediction:")[-1].strip()
            else:
                explanation = response_split[0].split("Explanation: ", 1)[1].strip()
                answer_text = response_split[-1].split("Prediction: ", 1)[1].strip()

            answer_parts = [p.strip() for p in answer_text.split(", ") if p.strip()]
            answer = {
                answer_parts[0].split(": ")[0]: parse_float(answer_parts[0].split(": ")[1]),
                answer_parts[1].split(": ")[0]: parse_float(answer_parts[1].split(": ")[1]),
            }
            return answer, explanation
        except Exception as e:
            print("exception: ", e)
            import pdb; pdb.set_trace()
            raise Exception(f"Error in parsing response {response}")
        

    if isinstance(image, list):
        responses = llm([(prompt,) + (massmap_to_pil_norm(i),) for i in image])
        return [process_response(response) for response in responses]

    else:
        response = llm([(prompt,) + (massmap_to_pil_norm(image),)])[0]
        return process_response(response)
  

def isolate_individual_features(
    explanation: str | list[str],
    model: str = "gpt-4o",
) -> list[str]:
    """
    Isolate individual features from the explanation by breaking it down into atomic claims.

    Args:
        explanation (str): The explanation text to break down into claims
        model (str): The OpenAI model to use for processing

    Returns:
        list[str]: A list of atomic claims extracted from the explanation
    """

    llm = load_model(model)

    if isinstance(explanation, list):
        prompts = [decomposition_massmaps.format(e) for e in explanation]
        results = llm(prompts)
        # print("isolate_individual_features results: ", results)
        all_all_claims: list[list[str]] = [
            [c.strip() for c in result.split("\n") if c.strip()]
            for result in results
        ]
        return all_all_claims
    else:
        raw_output = llm(decomposition_massmaps.format(explanation))
        # print("isolate_individual_features raw_output: ", raw_output)
        all_claims = [c.strip() for c in raw_output.split("\n") if c.strip()]
        return all_claims


def distill_relevant_features(
    example_image: PIL.Image.Image | torch.Tensor | np.ndarray,
    answer: str,
    atomic_claims: list[str],
    model: str = "gpt-4o",
    verbose: bool = False,
    massmap_to_pil_norm: Callable = massmap_to_pil_norm,
) -> list[str]:
    """
    Distill the relevant features from the atomic claims.
    """

    prompts = [load_relevance_massmaps_prompt(
        massmap_to_pil_norm(example_image), 
        f"Omega_m = {answer['Omega_m']}, sigma_8 = {answer['sigma_8']}",
        claim
    ) for claim in atomic_claims]
    llm = load_model(model)
    llm.verbose = True
    results = llm(prompts)

    if verbose:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(massmap_to_pil_norm(example_image))
        plt.show()
        import pprint
        print('atomic_claims')
        pprint.pprint(atomic_claims)
        print('results')
        pprint.pprint(results)

    relevant_claims = [
        claim for claim, result in zip(atomic_claims, results)
        if "relevance: yes" in result.lower()
    ]

    return relevant_claims


def get_claims_by_category(category: str, claims: list[str], model: str = "gpt-4o", verbose: bool = False):
    """
    Args:
        category (str): The category to find claims for.
        claims (list[str]): A list of relevant claims.
    Returns:
        dict: {"related_claims": list[str], "reasoning": str}
    """
    prompt = claim_grouping_massmaps.format(
        f'{category}: {category_mapping_massmaps["name2description"][category]}',
        '\n'.join(claims)
    )
    llm = load_model(model)
    response = llm([(prompt,)])[0].replace("\n\n", "\n")
    if response == "ERROR" or response is None or response == "":
        print("Error in querying OpenAI API")
        return None
    if verbose:
        print('===============================================')
        print("GETTING CLAIMS BY CATEGORY")
        print('category: ', category)
        print('claims: ', claims)
        print('response:', response)
        print('===============================================')
    # Extract GROUPED CLAIMS and REASONING sections using split on markers
    related_claims = []
    reasoning = ""

    # Normalize response for splitting
    response_sections = response
    if isinstance(response_sections, str):
        response_sections = response_sections.strip()

        # Split using markers "RELATED CLAIMS:" and "REASONING:"
        parts = response_sections.split("RELATED CLAIMS:")
        if len(parts) > 1:
            relevant_part = parts[1]
        else:
            relevant_part = parts[0]

        rel_claims, reasoning_raw = "", ""
        if "REASONING:" in relevant_part:
            rel_claims, reasoning_raw = relevant_part.split("REASONING:", 1)
        else:
            rel_claims = relevant_part
            reasoning_raw = ""

        # Related claims split by line, strip, ignore "n/a" & empty
        related_claims = [
            line.strip() for line in rel_claims.splitlines()
            if line.strip() and line.strip().lower() != "n/a"
        ]
        # If after split by lines the only thing left is a single empty string, convert to empty list
        if related_claims == [""]:
            related_claims = []

        reasoning = reasoning_raw.strip() if reasoning_raw else ""
    else:
        related_claims = []
        reasoning = ""

    return {
        "related_claims": related_claims,
        "reasoning": reasoning
    }

def group_claims_by_category(relevant_claims: list[str], model: str = "gpt-4o", verbose: bool = False):
    """
    Args:
        relevant_claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        dict[str, list[str]]: A dictionary where the keys are the categories and the values are lists of claims that are aligned with the category.
    """
    claims_by_category = {}
    for category in categories_list:
        claim_grouping_info = get_claims_by_category(category, relevant_claims, model, verbose)
        
        if claim_grouping_info is None or claim_grouping_info["related_claims"] is None:
            claims_by_category[category] = []
            continue

        related_claims = claim_grouping_info["related_claims"]
        reasoning = claim_grouping_info["reasoning"]
        if verbose:
            print('category: ', category)
            print('related_claims: ', related_claims)
            print('reasoning: ', reasoning)
        claims_by_category[category] = related_claims
    return claims_by_category

def calculate_expert_alignment_score_for_category(category: str, claims: list[str], model: str = "gpt-4o", verbose: bool = False):
    """
    Args:
        category (str): The category to calculate the alignment score for.
        claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        float: The alignment score for the claims in the category.
    """
    if len(claims) == 0:
        return {
            "alignment_label": "none",
            "alignment_score": 0.0,
            "reasoning": "No claims provided"
        }
    prompt = category_alignment_massmaps.format(
        f'{category}: {category_mapping_massmaps["name2description"][category]}', 
        '\n'.join(claims) if isinstance(claims, list) and len(claims) > 0 else 'N/A'
        )
    llm = load_model(model)
    response = llm([(prompt,)])[0].replace("\n\n", "\n")
    if response == "ERROR" or response is None or response == "":
        print("Error in querying OpenAI API")
        return None
    if verbose:
        print('===============================================')
        print("expert alignment score for category: ", category)
        print('response: ', response)
    # Separate out the alignment rating and reasoning
    alignment_mapping = {
        "complete": 1.0,
        "partial": 0.5,
        "none": 0.0,
    }

    # We'll extract lines for "Category Alignment Rating:" and "Reasoning:" and also allow fallback if not found
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]

    category_alignment = None
    reasoning = None

    for ln in lines:
        if ln.lower().startswith("category alignment rating:"):
            cat_rating_part = ln.split(":", 1)[1] if ":" in ln else ""
            category_alignment = cat_rating_part.strip().lower()
        elif ln.lower().startswith("reasoning:"):
            reasoning = ln.split(":", 1)[1].strip() if ":" in ln else ""

    # Fallback: If no alignment specified, look for the first nonempty line and treat that as alignment label
    if category_alignment is None and lines:
        category_alignment = lines[0].strip().lower()
    if reasoning is None:
        # Try to find a line that contains "reason"
        for ln in lines:
            if "reason" in ln.lower():
                reasoning = ln.split(":", 1)[1].strip() if ":" in ln else ""
                break

    # Map alignment label to score (try conversion if it's somehow numeric)
    score = None
    try:
        score = float(category_alignment)
    except Exception:
        score = alignment_mapping.get(category_alignment, 0.0)

    # Return a dictionary for compatibility
    return {
        "alignment_label": category_alignment,
        "alignment_score": score,
        "reasoning": reasoning
    }


def calculate_expert_alignment_score(claims: list[str], model: str = "gpt-4o", verbose: bool = False):
    claims_by_category = group_claims_by_category(claims, model, verbose)
    category_alignment_scores = {}
    category_alignment_reasonings = {}

    for category in categories_list:
        category_alignment_info = calculate_expert_alignment_score_for_category(category, claims_by_category[category], model, verbose)
        category_alignment_score = category_alignment_info["alignment_score"]
        category_alignment_reasoning = category_alignment_info["reasoning"]
        if category_alignment_score is None:
            raise Exception("Error in calculating expert alignment score for category: {}".format(category))
        category_alignment_scores[category] = category_alignment_score
        category_alignment_reasonings[category] = category_alignment_reasoning
    return claims_by_category, category_alignment_scores, category_alignment_reasonings


def make_alignment_matrix(claims, claims_by_category, category_alignment_scores):
    """
    Args:
        # categories (list[str]): A list of all expert categories.
        claims (list[str]): A list of all atomic claims.
        claims_by_category (dict[str, list[str]]): A dictionary where the keys are the categories and the values are lists of claims that are aligned with the category.
        category_alignment_scores (dict[str, float]): A dictionary where the keys are the categories and the values are the alignment scores.
    Returns:
        list[list[float]]: A matrix of alignment scores for the claims in the categories.
    """
    categories = categories_list
    matrix = np.zeros((len(claims), len(categories)))
    for i, claim in enumerate(claims):
        for j, category in enumerate(categories):
            if claim in claims_by_category[category]:
                matrix[i, j] = category_alignment_scores[category]
    return matrix
    
def calculate_expert_alignment_scores_old(
    claims: list[str],
    model: str = 'gpt-4o',
) -> list[dict]:
    """
    Parses LLM responses to extract:
      - Category
      - Category ID
      - Alignment (mapped from complete/partial/none)
      - Reasoning
    while ignoring any extra/noisy lines (e.g., "Output 4:").
    """

    llm = load_model(model)
    prompts = [alignment_massmaps.replace("[[CLAIM]]", claim) for claim in claims]
    responses = llm(prompts)

    # Accept a few reasonable label variants
    KEY_ALIASES = {
        "category": {"category"},
        "category_id": {"category id", "categoryID", "category_id"},
        "alignment": {"alignment", "category alignment", "category alignment rating", "alignment rating"},
        "reasoning": {"reasoning", "rationale", "explanation"},
    }

    alignment_mapping = {
        "complete": 1,
        "partial": 0.5,
        "none": 0,
    }

    # Helper: normalize keys and see which field it maps to
    def _which_field(k: str) -> str | None:
        k_norm = k.strip().lower()
        for field, aliases in KEY_ALIASES.items():
            if k_norm in aliases:
                return field
        return None

    # Regex to split "Key: Value" OR "Key - Value" (first separator only)
    kv_re = re.compile(r"^\s*([^:\-]+)\s*[:\-]\s*(.+?)\s*$")

    results = []
    for i, response in enumerate(responses):
        text = "" if response is None else str(response)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        category = None
        category_id = -1
        alignment_raw = None
        reasoning = None

        for ln in lines:
            m = kv_re.match(ln)
            if not m:
                # ignore lines that don't look like "Key: Value"
                continue
            key, value = m.group(1), m.group(2)

            field = _which_field(key)
            if field is None:
                # unrecognized key -> ignore
                continue

            if field == "category":
                category = value.strip()

            elif field == "category_id":
                # extract first integer if present; else leave -1
                m_id = re.search(r"-?\d+", value)
                if m_id:
                    try:
                        category_id = int(m_id.group(0))
                    except Exception:
                        category_id = -1

            elif field == "alignment":
                alignment_raw = value.strip()

            elif field == "reasoning":
                reasoning = value.strip()

        # Only add an entry if we at least got an alignment string
        if alignment_raw is not None:
            align_score = alignment_mapping.get(alignment_raw.lower(), 0)
            results.append({
                "Claim": claims[i],
                "Category": category if category is not None else "",
                "Category ID": category_id,
                "Alignment": align_score,
                "Alignment Raw": alignment_raw,
                "Reasoning": reasoning if reasoning is not None else "",
            })
        # else: ignore this response entirely (no alignment parsed)

    return results


def run_massmaps_generation(
    model: str = "gpt-4o",
    method: str = "vanilla",
    verbose: bool = False,
    overwrite_existing: bool = False,
    num_samples: int = 100,
    debug: bool = False,
) -> list[MassMapsExample]:
    """
    Runs the massmaps generation pipeline.
    """
    import os
    from pathlib import Path
    from tqdm.auto import tqdm

    import torch
    from datasets import load_dataset

    test_dataset = load_dataset("BrachioLab/massmaps-cosmogrid-100k", split='test')
    test_dataset.set_format('torch', columns=['input', 'label'])
    # Root dir is parent of parent of this file
    root_dir = Path(__file__).resolve().parent.parent
    save_dir = root_dir / "notebooks" / f"_dump/massmaps/intermediate/{model}/{method}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting to save intermediate results to {save_dir}")

    for idx in tqdm(range(num_samples)):
        save_path = os.path.join(save_dir, f'{idx}.json')
        if os.path.exists(save_path) and not overwrite_existing:
            continue
        X, y = test_dataset[idx:idx+1]['input'], test_dataset[idx:idx+1]['label']
        image = X[0]
        label = y[0]
        llm_answer, llm_explanation = get_llm_generated_answer(
            image, 
            method=method, 
            model=model
        )
        if llm_answer is None:
            continue
        example = MassMapsExample(
            input=image,
            answer={"Omega_m": label[0].item(), "sigma_8": label[1].item()},
            llm_answer=llm_answer,
            llm_explanation=llm_explanation
        )
        example.idx = idx

        # save intermediate
        save_dict = {}
        for k, v in example.__dict__.items():
            save_dict[k] = v if not isinstance(v, torch.Tensor) else v.cpu().numpy().tolist()
        with open(save_path, 'wt') as output_file:
            json.dump(save_dict, output_file)

def load_and_evaluate_massmaps_generation(
    model: str = "gpt-4o",
    method: str = "vanilla",
    verbose: bool = False,
    overwrite_existing: bool = False,
    num_samples: int = 100,
    debug: bool = False,
    eval_model: str = "gpt-5-mini-2025-08-07",
) -> list[MassMapsExample]:
    """
    Loads and evaluates the massmaps generation pipeline.
    """
    import os
    from pathlib import Path
    from tqdm.auto import tqdm

    import torch
    from datasets import load_dataset

    test_dataset = load_dataset("BrachioLab/massmaps-cosmogrid-100k", split='test')
    test_dataset.set_format('torch', columns=['input', 'label'])

    root_dir = Path(__file__).resolve().parent.parent
    load_dir = root_dir / "notebooks" / f"_dump/massmaps/intermediate/{model}/{method}"
    filenames = sorted([f for f in os.listdir(load_dir) if f.endswith('.json')], key=lambda x: int(x.split('.')[0]))
    all_results = []
    for filename in tqdm(filenames):
        path = os.path.join(load_dir, filename)
        with open(path, 'rt') as input_file:
            data = json.load(input_file)
        all_results.append(data)
    
    save_dir = root_dir / "notebooks" / f"_dump/massmaps/final/{model}/{method}/eval.{eval_model}"
    os.makedirs(save_dir, exist_ok=True)
    for idx in tqdm(range(len(all_results))):
        if idx >= num_samples:
            break
        save_path = os.path.join(save_dir, filenames[idx])
        if os.path.isfile(save_path):
            print('save_path', save_path)
            continue

        # load
        example_dict = all_results[idx]

        if not isinstance(example_dict['input'], torch.Tensor):
            example_dict['input'] = torch.tensor(example_dict['input'])

        example = MassMapsExample(
            input = example_dict['input'],
            answer = example_dict['answer'],
            llm_answer = example_dict['llm_answer'],
            llm_explanation = example_dict['llm_explanation'],
        )
        example.__dict__ = example_dict
        example.idx = example_dict['idx']

        # isolate individual features
        claims = isolate_individual_features(example.llm_explanation, model=eval_model)
        # print('claims', claims)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]

        # distill relevant features
        relevant_claims = distill_relevant_features(
            example.input, 
            example.llm_answer,
            example.claims,
            model=eval_model
        )
        example.relevant_claims = relevant_claims

        # calculate expert alignment scores
        claims_by_category, category_alignment_scores, category_alignment_reasonings = calculate_expert_alignment_score(
            relevant_claims, 
            eval_model,
            # verbose=True
        )

        example.claims_by_category = claims_by_category
        example.category_alignment_scores = category_alignment_scores
        example.category_alignment_reasonings = category_alignment_reasonings

        alignment_matrix = make_alignment_matrix(
            example.claims,
            claims_by_category,
            category_alignment_scores
        )
        
        final_alignment_score = alignment_matrix.max(axis=-1).mean()
        if np.isnan(final_alignment_score):
            print(f'example {idx} final_alignment_score is NaN')
        example.final_alignment_score = final_alignment_score

        # save
        save_dict = {}
        for k, v in example.__dict__.items():
            save_dict[k] = v if not isinstance(v, torch.Tensor) else v.cpu().numpy().tolist()
        with open(save_path, 'wt') as output_file:
            json.dump(save_dict, output_file)

def run_massmaps_pipeline(
    model: str = "gpt-4o", 
    method: str = "vanilla", 
    verbose: bool = False, 
    overwrite_existing: bool = False, 
    num_samples: int = 100, 
    debug: bool = False,
    run_generation: bool = True,
    run_evaluation: bool = True,
    eval_model: str = "gpt-5-mini-2025-08-07",
):
    """
    Runs the massmaps generation pipeline.
    """
    if run_generation:
        run_massmaps_generation(model, method, verbose, overwrite_existing, num_samples, debug)
    if run_evaluation:
        load_and_evaluate_massmaps_generation(model, method, verbose, overwrite_existing, num_samples, debug, eval_model)


def aggregate_all_results(
    models=[
        "gpt-5.2-pro-2025-12-11",
        "gpt-5-mini-2025-08-07",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "gemini-2.5-pro",
        "gemini-2.5-flash"
    ],
    eval_model="gpt-5-mini-2025-08-07",
    num_samples=100,
):
    """
    Aggregate/compare results across models/methods, compute mse, and save combined outputs.
    """
    # Adjust results_dir to notebooks/_dump/massmaps/final by default, relative to root_dir
    root_dir = Path(__file__).resolve().parent.parent
    final_results_dir = root_dir / "notebooks" / "_dump" / "massmaps" / "final"
    aggregated_results_dir = root_dir / "results"
    
    models = [
        "gpt-5.2-pro-2025-12-11",
        "gpt-5-mini-2025-08-07",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "gemini-2.5-pro",
        "gemini-2.5-flash"
    ]

    methods = [
        'vanilla', 
        'cot', 
        'socratic', 
        'subq'
    ]

    aggregated_paths = []
    loaded_dirs = []

    for model in models:
        filenames_per_method = {}
        for method in tqdm(methods, desc=f"Loading filenames for {model}"):
            load_dir = os.path.join(final_results_dir, model, method, f"eval.{eval_model}")
            if not os.path.isdir(load_dir):
                print(f"Warning: {load_dir} does not exist, skipping.")
                filenames_per_method[method] = set()
                continue
            filenames = set(os.listdir(load_dir))
            filenames_per_method[method] = filenames

        # Instead of using intersection, just keep the actual filenames for each method
        for method in tqdm(methods, desc=f"Sorting filenames for {model}"):
            file_list = [fn for fn in filenames_per_method[method] if fn.endswith('.json')]
            file_list = sorted(file_list)[:num_samples]
            filenames_per_method[method] = file_list

        all_results = defaultdict(list)
        for method in tqdm(methods, desc=f"Loading results for {model}"):
            load_dir = os.path.join(final_results_dir, model, method, f"eval.{eval_model}")
            loaded_dirs.append(load_dir)
            for filename in tqdm(filenames_per_method[method], desc=f"Loading results for {model} {method}"):
                path = os.path.join(load_dir, filename)
                if not os.path.exists(path):
                    print(f"Missing file {path}, skipping.")
                    continue
                with open(path, 'rt') as input_file:
                    data = json.load(input_file)
                # Compute MSE for specific fields if present
                mse_omega_m = None
                mse_sigma_8 = None
                try:
                    answer = data.get('answer', {})
                    llm_answer = data.get('llm_answer', {})
                    omega_true = answer.get('Omega_m')
                    omega_pred = llm_answer.get('Omega_m')
                    sigma_true = answer.get('sigma_8')
                    sigma_pred = llm_answer.get('sigma_8')
                    if omega_true is not None and omega_pred is not None:
                        mse_omega_m = mean_squared_error([omega_true], [omega_pred])
                    if sigma_true is not None and sigma_pred is not None:
                        mse_sigma_8 = mean_squared_error([sigma_true], [sigma_pred])
                    if mse_omega_m is not None or mse_sigma_8 is not None:
                        data['mse_loss'] = {'Omega_m': mse_omega_m, 'sigma_8': mse_sigma_8}
                except Exception as e:
                    print(f"Error computing MSE for {path}: {e}")
                data["_filename"] = filename
                all_results[method].append(data)
        
        for method in tqdm(all_results, desc=f"Saving results for {model}"):
            save_dir = os.path.join(aggregated_results_dir, method)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'massmaps_{model}_{eval_model}.json')
            save_path2 = os.path.join(save_dir, f'massmaps_{model}.json')
            with open(save_path, 'wt') as output_file:
                json.dump(all_results[method], output_file, indent=4)
            aggregated_paths.append(save_path)
            if eval_model == "gpt-5-mini-2025-08-07":
                with open(save_path2, 'wt') as output_file:
                    json.dump(all_results[method], output_file, indent=4)
                aggregated_paths.append(save_path2)
            print(f"Saved: {save_path}")

    print("====")
    print("Loaded directories:")
    for dir_path in set(loaded_dirs):
        print(dir_path)

    print("----")
    print("Paths of aggregated result files:")
    for path in aggregated_paths:
        print(path)
 

def parse_args():
    parser = argparse.ArgumentParser(description="Massmaps Generation and Evaluation Pipeline")

    subparsers = parser.add_subparsers(dest="command", help="Mode of operation", required=True)

    # Generation/evaluation parser
    gen_eval_parser = subparsers.add_parser("run", help="Run generation/evaluation for a model/method")
    group_run = gen_eval_parser.add_argument_group("Run Generation/Evaluation")
    group_run.add_argument('--model', type=str, default="gpt-4o", help="Model name (e.g. gpt-4o, gpt-4, etc)")
    group_run.add_argument('--method', type=str, default="vanilla", help="Method (e.g. vanilla, xyz)")
    group_run.add_argument('--verbose', action="store_true", help="Verbose output")
    group_run.add_argument('--overwrite_existing', action="store_true", help="Overwrite existing results")
    group_run.add_argument('--num_samples', type=int, default=100, help="Number of samples to process")
    group_run.add_argument('--debug', action="store_true", help="Debug mode (silent try/except during evaluation)")
    group_run.add_argument('--run_generation', action="store_true", help="Run generation step")
    group_run.add_argument('--run_evaluation', action="store_true", help="Run evaluation step")
    group_run.add_argument('--eval_model', type=str, default="gpt-5-mini-2025-08-07", help="Evaluation model name (e.g. gpt-5-mini-2025-08-07, gpt-4o, etc)")
    group_run.set_defaults(run_generation=False, run_evaluation=False)

    # Aggregate parser
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate all available result JSONs")
    group_agg = agg_parser.add_argument_group("Aggregate Results")
    group_agg.add_argument('--num_samples', type=int, default=100, help="Number of samples to aggregate")
    group_agg.add_argument('--eval_model', type=str, default="gpt-5-mini-2025-08-07", help="Evaluation model name (e.g. gpt-5-mini-2025-08-07, gpt-4o, etc)")

    return parser.parse_args()

def load_api_keys(root_dir):
    import json
    with open(f"{root_dir}/API_KEYS2.json", "r") as file:
        api_keys = json.load(file)
    os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
    os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']
    # os.environ['GOOGLE_API_KEY'] = api_keys['GOOGLE_API_KEY']
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(root_dir, api_keys['GOOGLE_APPLICATION_CREDENTIALS'])
    os.environ['CACHE_DIR'] = os.path.join(root_dir, 'cache_dir3')
    return api_keys

if __name__ == "__main__":
    """
    Example command lines for running the script in generation-only or evaluation-only mode:

    To run only generation for model "gpt-5.2-pro-2025-12-11" and method "vanilla":
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method vanilla --run_generation

    To run only evaluation (no generation) for model "gpt-5.2-pro-2025-12-11" and method "vanilla":
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method vanilla --run_evaluation

    Example command line for running the script with chain-of-thought (cot) reasoning enabled:

    To run generation for model "gpt-5.2-pro-2025-12-11" and method "cot":
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method cot --run_generation

    To run both generation and evaluation for model "gpt-5.2-pro-2025-12-11" and method "cot" with 50 samples:
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method cot --run_generation --run_evaluation --num_samples 50 --eval_model gpt-5-mini-2025-08-07

    To run both generation and evaluation for model "gpt-5.2-pro-2025-12-11" and method "vanilla":
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method vanilla --run_generation --run_evaluation --eval_model gpt-5-mini-2025-08-07

    Example command lines for running the script in generation-only or evaluation-only mode with different num_samples:

    To run only generation for model "gpt-5.2-pro-2025-12-11" and method "vanilla" with 5 samples:
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method vanilla --run_generation --num_samples 5

    To run only evaluation (no generation) for model "gpt-5.2-pro-2025-12-11" and method "vanilla" with 10 samples:
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method vanilla --run_evaluation --num_samples 10

    To run both generation and evaluation for model "gpt-5.2-pro-2025-12-11" and method "vanilla" using 100 samples:
      python src/massmaps.py run --model gpt-5.2-pro-2025-12-11 --method vanilla --run_generation --run_evaluation --eval_model gpt-5-mini-2025-08-07 --num_samples 100
    
    # Example command line for aggregation mode:
    #
    # To aggregate results from all models and methods (default settings):
    #   python src/massmaps.py aggregate
    #
    # To aggregate with custom result directories:
    #   python src/massmaps.py aggregate --final_results_dir notebooks/_dump/massmaps/final --aggregated_results_dir results
    #
    # To aggregate only a limited number of examples (e.g., 25 examples):
    #   python src/massmaps.py aggregate --num_samples 25
    """
    # Uncomment line below to install exlib
    # !pip install diskcache
    import sys; 
    sys.path.append('../src')

    ROOT_DIR = Path(__file__).resolve().parent.parent

    import openai
    import os

    load_api_keys(ROOT_DIR)


    args = parse_args()

    if args.command == "aggregate":
        
        aggregate_all_results(
            eval_model=args.eval_model,
            num_samples=args.num_samples,
        )

    elif args.command == "run":
        os.environ['LLMS_CACHE_PATH'] = os.path.join(ROOT_DIR, f'{args.model}.{args.method}.llms.py.cache')
        # Determine what to run based on CLI flags
        if not (args.run_generation or args.run_evaluation):
            print("No operation specified. Use --run_generation and/or --run_evaluation. See --help.")
            exit(1)
        run_massmaps_pipeline(
            model=args.model,
            method=args.method,
            verbose=args.verbose,
            overwrite_existing=args.overwrite_existing,
            num_samples=args.num_samples,
            debug=args.debug,
            run_generation=args.run_generation,
            run_evaluation=args.run_evaluation,
            eval_model=args.eval_model,
        )
