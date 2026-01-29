import os
import re
import io
import json
import time
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Callable

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
import PIL

from datasets import load_dataset
from tqdm import tqdm

from openai import OpenAI
from llms import load_model

from prompts.explanations import sepsis_prompt, vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline
from prompts.claim_decomposition import decomposition_sepsis
from prompts.relevance_filtering import relevance_sepsis
from prompts.expert_alignment import alignment_sepsis
from prompts.category_mapping import category_mapping_sepsis
from prompts.claim_grouping import claim_grouping_sepsis
from prompts.expert_category_alignment import category_alignment_sepsis

from diskcache import Cache
cache = Cache("/shared_data0/chaenyk/llm_cache")

categories_list = [name for name, _ in sorted(category_mapping_sepsis["name2id"].items(), key=lambda x: x[1])]

class SepsisExample:
    def __init__(self,
                 time_series_text,
                 time_series_data: Dict[float, Dict[str, Union[float, str]]],
                 ground_truth: Any,
                 llm_label: Any,
                 llm_explanation: str):
        self.time_series_text = time_series_text
        self.time_series_data: Dict[float, Dict[str, Union[float, str]]] = time_series_data
        self.ground_truth = ground_truth
        self.llm_label = llm_label
        self.llm_explanation = llm_explanation
        self.claims = []
        self.relevant_claims = []
        self.alignment_scores = []
        self.alignment_categories = []
        self.alignment_reasonings = []

@cache.memoize()
def get_llm_output(prompt, images=None, model='gpt-5-nano'):
    """
    prompt: str
    images: list of PIL images
    system_prompt: str
    """

    llm = load_model(model)

    result = llm([(prompt, *images)])[0]
    # import pdb; pdb.set_trace()
    return result

_number_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def format_time_series_for_prompt(time_series_data: Dict[float, Dict[str, Union[float, str]]]) -> str:
    if not time_series_data:
        return "No time-series data provided."
    output_lines = []
    for time in sorted(time_series_data.keys()):
        output_lines.append(f"Time {time}:")
        measurements = time_series_data[time]
        if not measurements:
            output_lines.append("  (No measurements recorded at this time)")
            continue
        for name in sorted(measurements.keys()):
            value = measurements[name]
            value_repr = f"'{value}'" if isinstance(value, str) else str(value)
            output_lines.append(f"  {name}: {value_repr}")

    return "\n".join(output_lines)

def parse_measurement_string(data_string: str) -> Dict[float, Dict[str, Union[float, str]]]:
    measurements_by_time: Dict[float, Dict[str, Union[float, str]]] = {}
    if not isinstance(data_string, str) or not data_string.strip():
        return measurements_by_time 

    parts = [part.strip() for part in data_string.strip().rstrip(';').split(';') if part.strip()]
    for part in parts:
        time_str, measurement_part = part.split(':', 1)
        measurement_name, value_str = measurement_part.split(',', 1)

        time = float(time_str.strip())
        name = measurement_name.strip()
        value_str_cleaned = value_str.strip()
        try:
            value: Union[float, str] = float(value_str_cleaned)
        except ValueError:
            value = value_str_cleaned 

        if time not in measurements_by_time:
            measurements_by_time[time] = {}
        measurements_by_time[time][name] = value
    return measurements_by_time

def get_llm_generated_answer(time_series_data: Dict[float, Dict[str, Union[float, str]]], method: str = "vanilla", model="gpt-5-nano"):
    text = format_time_series_for_prompt(time_series_data)
    if method == "vanilla":
        prompt = sepsis_prompt.replace("[BASELINE_PROMPT]", vanilla_baseline).format(text)
    elif method == "cot":
        prompt = sepsis_prompt.replace("[BASELINE_PROMPT]", cot_baseline).format(text)
    elif method == "socratic":
        prompt = sepsis_prompt.replace("[BASELINE_PROMPT]", socratic_baseline).format(text)
    elif method == "least_to_most":
        prompt = sepsis_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline).format(text)
    else:
        raise ValueError(f"Invalid method: {method}")

    llm = load_model(model)
    
    response = llm(prompt)
    if response == "ERROR":
        print("Error in querying LLM")
        return None
    print(response)
    response_split = [e for e in response.split("\n") if (e != '' and e.split()[0] in ['Label:', 'Explanation:'])]
    llm_label = response_split[0].split("Label: ")[1].strip()
    explanation = response_split[1].split("Explanation: ")[1].strip()
    return llm_label, explanation

def isolate_individual_features(
    explanation: str | list[str],
    model: str = "gpt-5-nano",
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
        prompts = [decomposition_sepsis.format(e) for e in explanation]
        results = llm(prompts)
        all_all_claims: list[list[str]] = [
            [c.strip() for c in result.split("\n") if c.strip()]
            for result in results
        ]
        return all_all_claims
    else:
        raw_output = llm(decomposition_sepsis.format(explanation))
        all_claims = [c.strip() for c in raw_output.split("\n") if c.strip()]
        return all_claims

def is_claim_relevant(time_series_text, rating: str, claim: str, model: str = "gpt-5-nano"):
    prompt = relevance_sepsis.format(time_series_text, rating, claim)
    llm = load_model(model)
    response = llm(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Relevance:", "").strip()
    response = response.split("\n")
    relevance = response[0].strip()
    reasoning = response[1].replace("Reasoning:", "").strip()
    return relevance, reasoning

def distill_relevant_features(example: SepsisExample, model: str = "gpt-5-nano"):
    relevant_claims = []
    for claim in tqdm(example.claims):
        relevance, reasoning = is_claim_relevant(example.time_series_text, example.llm_label, claim, model=model)
        if relevance is None:
            continue
        if relevance == "Yes":
            relevant_claims.append(claim)
    return relevant_claims

def get_claims_by_category(category: str, claims: list[str], model: str = "gpt-4o", verbose: bool = False):
    """
    Args:
        category (str): The category to find claims for.
        claims (list[str]): A list of relevant claims.
    Returns:
        dict: {"related_claims": list[str], "reasoning": str}
    """
    prompt = claim_grouping_sepsis.format(
        f'{category} - {category_mapping_sepsis["name2description"][category]}',
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
    prompt = category_alignment_sepsis.format(
        f'{category} - {category_mapping_sepsis["name2description"][category]}', 
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


# def make_alignment_matrix(claims, claims_by_category, category_alignment_scores):
#     """
#     Args:
#         # categories (list[str]): A list of all expertcategories.
#         claims (list[str]): A list of all atomic claims.
#         claims_by_category (dict[str, list[str]]): A dictionary where the keys are the categories and the values are lists of claims that are aligned with the category.
#         category_alignment_scores (dict[str, float]): A dictionary where the keys are the categories and the values are the alignment scores.
#     Returns:
#         list[list[float]]: A matrix of alignment scores for the claims in the categories.
#     """
#     categories = categories_list
#     matrix = np.zeros((len(claims), len(categories)))
#     for i, claim in enumerate(claims):
#         for j, category in enumerate(categories):
#             if claim in claims_by_category[category]:
#                 matrix[i, j] = category_alignment_scores[category]
#     return matrix

def make_alignment_matrix(claims, claims_by_category, category_alignment_scores, threshold=90):
    """
    Args:
        claims (list[str]): A list of all atomic claims.
        claims_by_category (dict[str, list[str]]): keys=category, values=list of claims aligned with the category.
        category_alignment_scores (dict[str, float]): keys=category, values=alignment score.
        threshold (int): fuzzy match threshold (exclusive), default > 90 to match original behavior.
    Returns:
        np.ndarray: shape (len(claims), len(categories))
    """
    from fuzzywuzzy import fuzz
    import numpy as np

    categories = categories_list  # kept as in your original code

    # Optional normalization (uncomment if you want it)
    def norm(s: str) -> str:
        return s  # or: return s.strip("- ").lower()

    # ---- 1) Precompute matches for unique claims ----
    unique_claims = {norm(c) for c in claims}

    # Map: claim -> set(categories it matches)
    claim_match_map = {c: set() for c in unique_claims}

    for category in categories:
        cat_claims = [norm(x) for x in claims_by_category.get(category, [])]
        if not cat_claims:
            continue

        for c in unique_claims:
            # Match if ANY claim in that category is similar enough
            if any(fuzz.ratio(c, cc) > threshold for cc in cat_claims):
                claim_match_map[c].add(category)

    # ---- 2) Build matrix via lookup ----
    matrix = np.zeros((len(claims), len(categories)), dtype=float)

    for i, claim in enumerate(claims):
        c = norm(claim)
        for category in claim_match_map.get(c, ()):
            j = categories.index(category)  # small categories => fine; see note below
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
    prompts = [alignment_sepsis.replace("[[CLAIM]]", claim) for claim in claims]
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