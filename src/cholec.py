import os
import random
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Any
import argparse
import numpy as np
from collections import defaultdict
import torch
import PIL
import re
from torch.utils.data import Dataset
from torchvision import transforms as tfs
import datasets as hfds
from diskcache import Cache

# Local imports
from llms import load_model
from prompts.claim_decomposition import decomposition_cholec
from prompts.relevance_filtering import load_relevance_cholec_prompt
from prompts.expert_alignment import alignment_cholec
from prompts.explanations import load_cholec_prompt
from prompts.claim_grouping import claim_grouping_cholec
from prompts.expert_category_alignment import category_alignment_cholec
from prompts.category_mapping import category_mapping_cholec

default_model = "gpt-4o"

categories_list = [name for name, _ in sorted(category_mapping_cholec["name2id"].items(), key=lambda x: x[1])]

class CholecExample:
    def __init__(
        self,
        id: str,
        image: torch.Tensor,
        true_safe_list: list[int],
        true_unsafe_list: list[int],
        llm_raw_output: str,
        llm_explanation: str,
        llm_safe_list: list[int],
        llm_unsafe_list: list[int],
    ):
        """
        Args:
            id: The ID of the example from the HuggingFace dataset.
            image: The image of the gallbladder surgery.
            safe_list_ground_truth: The ground truth safe regions. (length = 9x16 = 144)
            unsafe_list_ground_truth: The ground truth unsafe regions. (length = 9x16 = 144)
            llm_explanation: The explanation of the safe/unsafe regions.
        """
        self.id = id
        self.image = image
        self.true_safe_list = true_safe_list
        self.true_unsafe_list = true_unsafe_list
        self.llm_raw_output = llm_raw_output
        self.llm_explanation = llm_explanation
        self.llm_safe_list = llm_safe_list
        self.llm_unsafe_list = llm_unsafe_list

        # All raw claims obtained from the LLM
        self.all_claims : list[str] = []

        # Claims that are relevant to the explanation
        self.relevant_claims : list[str] = []

        # Relevant claims for which the LLM successfully managed to make an alignment judgment.
        self.alignable_claims : list[str] = []
        self.aligned_category_ids : list[int] = [] # Same length as alignable claims
        self.alignment_scores : list[float] = [] # Same length as alignable claims
        self.alignment_reasonings : list[str] = [] # Same length as alignable claims

        # The final alignment score, computed as the mean of the alignment scores of the alignable claims.
        self.final_alignment_score : float = 0.0

        # The LLM's prediction of the safe/unsafe regions
        self.safe_iou : float = 0.0
        self.unsafe_iou : float = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "true_safe_list": self.true_safe_list,
            "true_unsafe_list": self.true_unsafe_list,
            "llm_raw_output": self.llm_raw_output,
            "llm_explanation": self.llm_explanation,
            "llm_safe_list": self.llm_safe_list,
            "llm_unsafe_list": self.llm_unsafe_list,
            "all_claims": self.all_claims,
            "relevant_claims": self.relevant_claims,
            "alignable_claims": self.alignable_claims,
            "aligned_category_ids": self.aligned_category_ids,
            "alignment_scores": self.alignment_scores,
            "alignment_reasonings": self.alignment_reasonings,
            "final_alignment_score": self.final_alignment_score,
            "safe_iou": self.safe_iou,
            "unsafe_iou": self.unsafe_iou,
        }

    def __str__(self):
        return self.to_dict().__str__()


class CholecDataset(Dataset):
    """
    The cholecystectomy (gallbladder surgery) dataset, loaded from HuggingFace.
    The task is to find the safe/unsafe (gonogo) regions.
    The expert-specified features are the organ labels.

    For more details, see: https://huggingface.co/datasets/BrachioLab/cholec
    """

    gonogo_names: str = ["Background", "Safe", "Unsafe"]
    organ_names: str = ["Background", "Liver", "Gallbladder", "Hepatocystic Triangle"]

    def __init__(
        self,
        split: str = "train",
        hf_data_repo: str = "BrachioLab/cholec",
        image_size: tuple[int] = (360, 640)
    ):
        """
        Args:
            split: The options are "train" and "test".
            hf_data_repo: The HuggingFace repository where the dataset is stored.
            image_size: The (height, width) of the image to load.
        """
        self.dataset = hfds.load_dataset(hf_data_repo, split=split)
        self.dataset.set_format("torch")
        self.image_size = image_size
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float() / 255),
            tfs.Resize(image_size),
        ])
        self.preprocess_labels = tfs.Compose([
            tfs.Lambda(lambda x: x.unsqueeze(0)),
            tfs.Resize(image_size),
            tfs.Lambda(lambda x: x[0])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset[idx]['image'].shape[-1] == 3: # and self.dataset[idx]['image'].shape[:2] == self.image_size:
            image = self.dataset[idx]['image'].permute(2,0,1)
        else:
            image = self.dataset[idx]['image']
        image = self.preprocess_image(image)
        gonogo = self.preprocess_labels(self.dataset[idx]["gonogo"]).long()
        organs = self.preprocess_labels(self.dataset[idx]["organ"]).long()
        return {
            "id": self.dataset[idx]["id"],
            "image": image,     # (3,H,W)
            "gonogo": gonogo,   # (H,W)
            "organs": organs,   # (H,W)
        }

def extract_list_from_string(text: str, list_name: str) -> list[int]:
    """
    Extracts an integer list from a string based on a list name marker.
    Handles lists enclosed in brackets or just comma-separated numbers.

    Args:
        text: The input string.
        list_name: The name preceding the list (e.g., "Safe List").

    Returns:
        A list of integers, or an empty list if the list is not found or parsed incorrectly.
    """
    # Regex to find the list pattern:
    # list_name, optional whitespace, :
    # followed by either:
    #   \[(.*?)\]  (content inside brackets)
    #   |          (OR)
    #   ([^.\n]+)  (any character except dot or newline, one or more times - captures the numbers)
    # We use re.DOTALL so . can match newlines if the list spans lines,
    # but the second part of the OR explicitly avoids newlines to stop capturing the list content.
    pattern = rf"(?i){re.escape(list_name)}[^:]*:\s*(?:\[(.*?)\]|([^.\n]+))"
    match = re.search(pattern, text, re.DOTALL)

    list_str = ""
    if match:
        # Check which group matched: group 1 for brackets, group 2 for raw numbers
        if match.group(1) is not None:
            list_str = match.group(1).strip()
        elif match.group(2) is not None:
            list_str = match.group(2).strip()

    if list_str:
        # Split by comma, strip whitespace, convert to int, filter out empty strings
        try:
            # Split by comma and optional whitespace around it
            return [int(x.strip()) for x in re.split(r',\s*', list_str) if x.strip()]
        except ValueError:
            # Handle cases where list content is not purely integers
            print(f"Warning: Could not parse list content for '{list_name}'. Returning empty list.")
            return []

    return [] # Return empty list if the pattern is not found or no content was captured

def extract_explanation_safe_unsafe(text: str) -> tuple[str, list[int], list[int]]:
    """
    Extracts explanation, safe_list, and unsafe_list from the raw string.
    Handles lists enclosed in brackets or just comma-separated numbers.

    Args:
        text: The input raw string.

    Returns:
        A tuple containing (explanation, safe_list, unsafe_list).
    """
    # Extract lists first
    safe_list = extract_list_from_string(text, "Safe List")
    unsafe_list = extract_list_from_string(text, "Unsafe List")

    # Remove the list sections from the text to get the explanation
    # Regex to match either bracketed list or comma-separated numbers
    list_pattern = r"{}:\s*(?:\[.*?\]|[^.\n]+)".format(re.escape("Safe List"))
    explanation = re.sub(list_pattern, "", text, flags=re.DOTALL)

    list_pattern = r"{}:\s*(?:\[.*?\]|[^.\n]+)".format(re.escape("Unsafe List"))
    explanation = re.sub(list_pattern, "", explanation, flags=re.DOTALL)


    # Clean up extra newlines that might result from removing the lists
    explanation = explanation.strip()
    # Replace multiple consecutive newlines with at most two
    explanation = re.sub(r'\n\s*\n', '\n\n', explanation)

    return explanation, safe_list, unsafe_list


def get_llm_generated_answer(
    image: torch.Tensor | np.ndarray | PIL.Image.Image | list[Any],
    model: str = default_model,
    baseline: str = "vanilla",
) -> dict[str, Any]:
    """
    Generate a detailed surgical analysis and segmentation masks using an LLM.
    
    This function sends a surgical image to an LLM and receives back:
    1. A detailed explanation of safe/unsafe regions
    2. Binary masks for safe/unsafe regions
    
    Args:
        image: Input surgical image in tensor, numpy array, or PIL Image format
        model: Name of the LLM model to use (default: "gpt-4o")
        baseline: The baseline to use for the explanation (default: "vanilla")
            Options: "vanilla", "cot", "socratic", "least_to_most"
        
    Returns:
        Dictionary containing:
            - "Answer": The description of where it is safe and unsafe to operate
            - "Explanation": Detailed text analysis of safe/unsafe regions
    """

    llm = load_model(model)

    prompt = load_cholec_prompt(baseline)

    if isinstance(image, list):
        prompts = [prompt + (i,) for i in image]
        responses = llm(prompts)
        return responses

    else:
        response = llm(prompt + (image,))
        return response


def isolate_individual_features(
    explanation: str | list[str],
    model: str = default_model,
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
        prompts = [decomposition_cholec.format(e) for e in explanation]
        results = llm(prompts)
        all_all_claims: list[list[str]] = [
            [c.strip() for c in result.split("\n") if c.strip()]
            for result in results
        ]
        return all_all_claims
    else:
        raw_output = llm(decomposition_cholec.format(explanation))
        all_claims = [c.strip() for c in raw_output.split("\n") if c.strip()]
        return all_claims


def distill_relevant_features(
    example_image: PIL.Image.Image | torch.Tensor | np.ndarray,
    atomic_claims: list[str],
    model: str = default_model,
) -> list[str]:
    """
    Distill the relevant features from the atomic claims.
    """

    prompts = [load_relevance_cholec_prompt(example_image, claim) for claim in atomic_claims]
    llm = load_model(model)
    llm.verbose = True
    results = llm(prompts)

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
    prompt = claim_grouping_cholec.format(
        f'{category} - {category_mapping_cholec["name2description"][category]}',
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
    if len(claims) == 0:
        return {
            "alignment_label": "none",
            "alignment_score": 0.0,
            "reasoning": "No claims provided"
        }
    prompt = category_alignment_cholec.format(
        f'{category} - {category_mapping_cholec["name2description"][category]}', 
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
        # categories (list[str]): A list of all expertcategories.
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
    prompts = [alignment_cholec.replace("[[CLAIM]]", claim) for claim in claims]
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



def _avg_pool_mask(mask, a2d):
    """
    Accepts mask of shape (H,W) or (C,H,W) or (N,C,H,W) (any dtype, incl. bool),
    pools it with AvgPool2d, and returns a 2D tensor (H', W').
    """
    t = torch.as_tensor(mask)
    # Normalize to (N, C, H, W)
    if t.ndim == 2:            # (H, W)
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:          # (C, H, W)
        t = t.unsqueeze(0)
    elif t.ndim == 4:          # (N, C, H, W)
        pass
    else:
        raise ValueError(f"Unsupported mask ndim: {t.ndim} (expected 2, 3, or 4)")

    out = a2d(t.float())
    # Squeeze back to (H', W')
    return out.squeeze(0).squeeze(0)


def items_to_examples_old(
    items: list[dict],
    explanation_model: str = default_model,
    evaluation_model: str = default_model,
    baseline: str = "vanilla",
    verbose: bool = False,
) -> list[CholecExample]:
    """
    Convert an image to a CholecExample by running the entire LLM pipeline.
    """
    _start_time = time.time()

    # Compute the true safe/unsafe lists
    grid_size = 40
    a2d = torch.nn.AvgPool2d(kernel_size=grid_size, stride=grid_size)

    true_safe_avgs = [(_avg_pool_mask((item["gonogo"] == 1).float(), a2d).squeeze() > 0.1).long() for item in items]
    true_unsafe_avgs = [(_avg_pool_mask((item["gonogo"] == 2).float(), a2d).squeeze() > 0.1).long() for item in items]

    true_safe_lists = [sr.view(-1).nonzero().view(-1).tolist() for sr in true_safe_avgs]
    true_unsafe_lists = [ur.view(-1).nonzero().view(-1).tolist() for ur in true_unsafe_avgs]

    # Step 0: Get the LLM answers
    _t = time.time()

    llm_answers = get_llm_generated_answer([item["image"] for item in items], explanation_model, baseline)
    if verbose:
        print(f"Time taken to get LLM answers: {time.time() - _t:.3f} seconds")

    llm_outs = [extract_explanation_safe_unsafe(llm_answer) for llm_answer in llm_answers]

    examples = [
        CholecExample(
            id=items[i]["id"],
            image=items[i]["image"],
            true_safe_list=true_safe_lists[i],
            true_unsafe_list=true_unsafe_lists[i],
            llm_raw_output=llm_answers[i],
            llm_explanation=llm_outs[i][0],
            llm_safe_list=llm_outs[i][1],
            llm_unsafe_list=llm_outs[i][2],
        )
        for i in range(len(items))
    ]

    # Step 0.5: Calculate the accuracy of the LLM's prediction of the safe/unsafe regions as an IOU score
    for i in range(len(items)):
        true_safes = set(true_safe_lists[i])
        true_unsafes = set(true_unsafe_lists[i])
        llm_safes = set(examples[i].llm_safe_list)
        llm_unsafes = set(examples[i].llm_unsafe_list)

        if len(true_safes) > 0:
            examples[i].safe_iou = len(true_safes & llm_safes) / len(true_safes | llm_safes)
        else:
            examples[i].safe_iou = 0.0

        if len(true_unsafes) > 0:
            examples[i].unsafe_iou = len(true_unsafes & llm_unsafes) / len(true_unsafes | llm_unsafes)
        else:
            examples[i].unsafe_iou = 0.0


    # Step 1: Decompose the LLM explanation into atomic claims
    _t = time.time()
    all_all_claims = isolate_individual_features([example.llm_explanation for example in examples], evaluation_model)
    if verbose:
        print(f"Time taken to decompose into atomic claims: {time.time() - _t:.3f} seconds")

    for i in range(len(all_all_claims)):
        examples[i].all_claims = all_all_claims[i]

    # Step 2: Distill the relevant features from the atomic claims
    _t = time.time()
    for example in tqdm(examples):
        example.relevant_claims = distill_relevant_features(example.image, example.all_claims, evaluation_model)
    if verbose:
        print(f"Time taken to distill relevant features: {time.time() - _t:.3f} seconds")

    # Step 3: Calculate the expert alignment scores
    _t = time.time()
    for example in tqdm(examples):
        align_infos = calculate_expert_alignment_scores_old(example.relevant_claims, evaluation_model)

        example.alignable_claims = [info["Claim"] for info in align_infos]
        example.aligned_category_ids = [info["Category ID"] for info in align_infos]
        example.alignment_scores = [info["Alignment"] for info in align_infos]
        example.alignment_reasonings = [info["Reasoning"] for info in align_infos]

        # Non-alignable claims are given a score of 0.0
        if len(align_infos) > 0:
            example.final_alignment_score = sum(info["Alignment"] for info in align_infos) / len(example.all_claims)
        else:
            example.final_alignment_score = 0.0

    if verbose:
        print(f"Time taken to calculate expert alignment scores: {time.time() - _t:.3f} seconds")

    if verbose:
        print(f"Total time taken: {time.time() - _start_time:.3f} seconds")

    return examples


def run_cholec_pipeline_old(
    items: list[dict],
    explanation_model: str = default_model,
    evaluation_model: str = default_model,
    baseline: str = "vanilla",
    verbose: bool = False,
    overwrite_existing: bool = False,
) -> list[CholecExample]:
    """
    Run the cholecystectomy pipeline on a list of items.
    """
    save_path = str(Path(__file__).parent / ".." / "results" / baseline / f"cholec_{explanation_model}.json")
    if os.path.exists(save_path) and not overwrite_existing:
        print(f"Results already exist at {save_path}. Set overwrite_existing=True to overwrite.")
        return

    examples = items_to_examples_old(items, explanation_model, evaluation_model, baseline, verbose)
    with open(save_path, "w") as f:
        json.dump([example.to_dict() for example in examples], f, indent=4)


def get_yes_no_confirmation(prompt):
    """
    Prompts the user with a yes/no question and returns True for yes, False for no.
    Keeps asking until a valid response is given.
    """
    while True:
        response = input(prompt + " (Y/n): ").lower().strip()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")


def run_cholec_generation(
    model: str = "gpt-4o",
    method: str = "vanilla",
    verbose: bool = False,
    overwrite_existing: bool = False,
    num_samples: int = 100,
    debug: bool = False,
) -> list[CholecExample]:
    """
    Runs the cholec generation pipeline.
    """
    from pathlib import Path
    from tqdm.auto import tqdm

    dataset = CholecDataset(split="test", image_size=(360, 640))

    root_dir = Path(__file__).resolve().parent.parent
    save_dir = root_dir / "notebooks" / f"_dump/cholec/intermediate/{model}/{method}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting to save intermediate results to {save_dir}")

    grid_size = 40
    a2d = torch.nn.AvgPool2d(kernel_size=grid_size, stride=grid_size)

    for idx in tqdm(range(num_samples)):
        save_path = os.path.join(save_dir, f"{idx}.json")
        if os.path.exists(save_path) and not overwrite_existing:
            continue

        item = dataset[idx]
        image = item["image"]

        llm_answers = get_llm_generated_answer(
            [image],
            baseline=method,
            model=model,
        )
        if not llm_answers:
            continue
        llm_explanation, llm_safe_list, llm_unsafe_list = extract_explanation_safe_unsafe(llm_answers[0])

        true_safe_avg = (_avg_pool_mask((item["gonogo"] == 1).float(), a2d).squeeze() > 0.1).long()
        true_unsafe_avg = (_avg_pool_mask((item["gonogo"] == 2).float(), a2d).squeeze() > 0.1).long()

        true_safe_list = true_safe_avg.view(-1).nonzero().view(-1).tolist()
        true_unsafe_list = true_unsafe_avg.view(-1).nonzero().view(-1).tolist()

        example = CholecExample(
            id=item["id"],
            image=image,
            true_safe_list=true_safe_list,
            true_unsafe_list=true_unsafe_list,
            llm_raw_output=llm_answers[0],
            llm_explanation=llm_explanation,
            llm_safe_list=llm_safe_list,
            llm_unsafe_list=llm_unsafe_list,
        )

        save_dict = {}
        for k, v in example.__dict__.items():
            if k == "image":
                continue
            save_dict[k] = v if not isinstance(v, torch.Tensor) else v.cpu().numpy().tolist()
        with open(save_path, "wt") as output_file:
            json.dump(save_dict, output_file)


def load_and_evaluate_cholec_generation(
    model: str = "gpt-4o",
    method: str = "vanilla",
    verbose: bool = False,
    overwrite_existing: bool = False,
    num_samples: int = 100,
    debug: bool = False,
    eval_model: str = "gpt-5-mini-2025-08-07",
) -> list[CholecExample]:
    """
    Loads and evaluates the cholec generation pipeline.
    """
    from pathlib import Path
    from tqdm.auto import tqdm

    dataset = CholecDataset(split="test", image_size=(360, 640))

    root_dir = Path(__file__).resolve().parent.parent
    load_dir = root_dir / "notebooks" / f"_dump/cholec/intermediate/{model}/{method}"
    if not load_dir.is_dir():
        print(f"Warning: {load_dir} does not exist, skipping.")
        return []

    filenames = sorted([f for f in os.listdir(load_dir) if f.endswith(".json")], key=lambda x: int(x.split(".")[0]))
    all_results = []
    for filename in tqdm(filenames):
        path = os.path.join(load_dir, filename)
        with open(path, "rt") as input_file:
            data = json.load(input_file)
        all_results.append(data)

    save_dir = root_dir / "notebooks" / f"_dump/cholec/final/{model}/{method}/eval.{eval_model}"
    os.makedirs(save_dir, exist_ok=True)
    for idx in tqdm(range(len(all_results))):
        if idx >= num_samples:
            break
        save_path = os.path.join(save_dir, filenames[idx])
        if os.path.isfile(save_path):
            continue

        example_dict = all_results[idx]
        dataset_idx = int(os.path.basename(save_path).split(".")[0])
        if dataset[dataset_idx]["id"] != example_dict["id"]:
            raise ValueError(
                "dataset[dataset_idx]['id'] != example_dict['id'], dataset[dataset_idx]['id'] is ",
                dataset[dataset_idx]["id"],
                " and example_dict['id'] is ",
                example_dict["id"],
            )

        example = CholecExample(
            id=example_dict["id"],
            image=dataset[dataset_idx]["image"],
            true_safe_list=example_dict["true_safe_list"],
            true_unsafe_list=example_dict["true_unsafe_list"],
            llm_raw_output=example_dict["llm_raw_output"],
            llm_explanation=example_dict["llm_explanation"],
            llm_safe_list=example_dict["llm_safe_list"],
            llm_unsafe_list=example_dict["llm_unsafe_list"],
        )

        claims = isolate_individual_features(example.llm_explanation, model=eval_model)
        if claims is None:
            continue
        example.all_claims = [claim.strip() for claim in claims]

        relevant_claims = distill_relevant_features(
            example.image,
            example.all_claims,
            model=eval_model,
        )
        example.relevant_claims = relevant_claims

        claims_by_category, category_alignment_scores, category_alignment_reasonings = calculate_expert_alignment_score(
            relevant_claims,
            eval_model,
        )

        example.claims_by_category = claims_by_category
        example.category_alignment_scores = category_alignment_scores
        example.category_alignment_reasonings = category_alignment_reasonings

        alignment_matrix = make_alignment_matrix(
            example.all_claims,
            claims_by_category,
            category_alignment_scores,
        )

        final_alignment_score = alignment_matrix.max(axis=-1).mean()
        example.final_alignment_score = final_alignment_score

        save_dict = {}
        for k, v in example.__dict__.items():
            if k == "image":
                continue
            save_dict[k] = v if not isinstance(v, torch.Tensor) else v.cpu().numpy().tolist()
        with open(save_path, "wt") as output_file:
            json.dump(save_dict, output_file)


def run_cholec_pipeline(
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
    Runs the cholec generation pipeline.
    """
    if run_generation:
        run_cholec_generation(model, method, verbose, overwrite_existing, num_samples, debug)
    if run_evaluation:
        load_and_evaluate_cholec_generation(model, method, verbose, overwrite_existing, num_samples, debug, eval_model)


def aggregate_all_results(
    models=[
        "gpt-5.2-pro-2025-12-11",
        "gpt-5-mini-2025-08-07",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
    eval_model="gpt-5-mini-2025-08-07",
    num_samples=100,
):
    """
    Aggregate/compare results across models/methods and save combined outputs.
    """
    root_dir = Path(__file__).resolve().parent.parent
    final_results_dir = root_dir / "notebooks" / "_dump" / "cholec" / "final"
    aggregated_results_dir = root_dir / "results"

    models = [
        "gpt-5.2-pro-2025-12-11",
        "gpt-5-mini-2025-08-07",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    methods = [
        "vanilla",
        "cot",
        "socratic",
        "subq",
    ]

    aggregated_paths = []
    loaded_dirs = []

    for model in models:
        filenames_per_method = {}
        for method in methods:
            load_dir = os.path.join(final_results_dir, model, method, f"eval.{eval_model}")
            if not os.path.isdir(load_dir):
                print(f"Warning: {load_dir} does not exist, skipping.")
                filenames_per_method[method] = set()
                continue
            filenames = set(os.listdir(load_dir))
            filenames_per_method[method] = filenames

        for method in methods:
            file_list = [fn for fn in filenames_per_method[method] if fn.endswith(".json")]
            file_list = sorted(file_list)[:num_samples]
            filenames_per_method[method] = file_list

        all_results = defaultdict(list)
        for method in tqdm(methods, desc=f"Aggregate-{model}"):
            load_dir = os.path.join(final_results_dir, model, method, f"eval.{eval_model}")
            loaded_dirs.append(load_dir)
            for filename in filenames_per_method[method]:
                path = os.path.join(load_dir, filename)
                if not os.path.exists(path):
                    print(f"Missing file {path}, skipping.")
                    continue
                with open(path, "rt") as input_file:
                    data = json.load(input_file)

                true_safes = set(data["true_safe_list"])
                true_unsafes = set(data["true_unsafe_list"])
                llm_safes = set(data["llm_safe_list"])
                llm_unsafes = set(data["llm_unsafe_list"])

                if len(true_safes) > 0:
                    data["safe_iou"] = len(true_safes & llm_safes) / len(true_safes | llm_safes)
                else:
                    data["safe_iou"] = 0.0

                if len(true_unsafes) > 0:
                    data["unsafe_iou"] = len(true_unsafes & llm_unsafes) / len(true_unsafes | llm_unsafes)
                else:
                    data["unsafe_iou"] = 0.0

                data["_filename"] = filename
                all_results[method].append(data)

        for method in all_results:
            save_dir = os.path.join(aggregated_results_dir, method)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"cholec_{model}_{eval_model}.json")
            save_path2 = os.path.join(save_dir, f"cholec_{model}.json")
            with open(save_path, "wt") as output_file:
                json.dump(all_results[method], output_file, indent=4)
            aggregated_paths.append(save_path)
            if eval_model == "gpt-5-mini-2025-08-07":
                with open(save_path2, "wt") as output_file:
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
    parser = argparse.ArgumentParser(description="Cholec Generation and Evaluation Pipeline")

    subparsers = parser.add_subparsers(dest="command", help="Mode of operation", required=True)

    gen_eval_parser = subparsers.add_parser("run", help="Run generation/evaluation for a model/method")
    group_run = gen_eval_parser.add_argument_group("Run Generation/Evaluation")
    group_run.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g. gpt-4o, gpt-4, etc)")
    group_run.add_argument("--method", type=str, default="vanilla", help="Method (e.g. vanilla, xyz)")
    group_run.add_argument("--verbose", action="store_true", help="Verbose output")
    group_run.add_argument("--overwrite_existing", action="store_true", help="Overwrite existing results")
    group_run.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    group_run.add_argument("--debug", action="store_true", help="Debug mode (silent try/except during evaluation)")
    group_run.add_argument("--run_generation", action="store_true", help="Run generation step")
    group_run.add_argument("--run_evaluation", action="store_true", help="Run evaluation step")
    group_run.add_argument("--eval_model", type=str, default="gpt-5-mini-2025-08-07", help="Evaluation model name")
    group_run.set_defaults(run_generation=False, run_evaluation=False)

    agg_parser = subparsers.add_parser("aggregate", help="Aggregate all available result JSONs")
    group_agg = agg_parser.add_argument_group("Aggregate Results")
    group_agg.add_argument("--num_samples", type=int, default=100, help="Number of samples to aggregate")
    group_agg.add_argument("--eval_model", type=str, default="gpt-5-mini-2025-08-07", help="Evaluation model name")

    return parser.parse_args()


def load_api_keys(root_dir):
    with open(f"{root_dir}/API_KEYS2.json", "r") as file:
        api_keys = json.load(file)
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = api_keys["ANTHROPIC_API_KEY"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(root_dir, api_keys["GOOGLE_APPLICATION_CREDENTIALS"])
    os.environ["CACHE_DIR"] = os.path.join(root_dir, "cache_dir3")
    return api_keys


if __name__ == "__main__":
    import sys

    sys.path.append("../src")

    root_dir = Path(__file__).resolve().parent.parent
    load_api_keys(root_dir)

    args = parse_args()

    if args.command == "aggregate":
        aggregate_all_results(
            eval_model=args.eval_model,
            num_samples=args.num_samples,
        )
    elif args.command == "run":
        if not (args.run_generation or args.run_evaluation):
            print("No operation specified. Use --run_generation and/or --run_evaluation. See --help.")
            exit(1)
        run_cholec_pipeline(
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
