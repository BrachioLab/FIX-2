import os
import random
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Any
import numpy as np
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
        image_size: tuple[int] = (180, 320)
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
        if self.dataset[idx]['image'].shape[:2] == self.image_size:
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


if __name__ == "__main__":
    _start_time = time.time()

    # Take a few random, unique samples from the dataset
    random.seed(42)
    num_samples = 150
    dataset = CholecDataset(split="test", image_size=(360, 640))
    random_indices = random.sample(range(len(dataset)), num_samples)
    print(f"Random indices: {random_indices}")
    items = [dataset[i] for i in random_indices]

    # models = ["gpt-4o", "o1", "claude-3-5-sonnet-latest", "gemini-2.5-pro-exp-03-25"]
    # models = ["gpt-4o", "o1", "claude-3-5-sonnet-latest", "gemini-2.0-flash"]
    models = ["gemini-2.0-flash"]
    baselines = ["vanilla", "cot", "socratic", "subq"]

    # Can be very expensive!
    if get_yes_no_confirmation("You are about to spend a lot of money"):
        # Run the models and baselines
        for model in models:
            _model_time = time.time()
            for baseline in baselines:
                print(f"\nRunning {model} with {baseline} baseline...")
                run_cholec_pipeline(
                    items=items,
                    explanation_model=model,
                    evaluation_model="gpt-4o",
                    baseline=baseline,
                    verbose=True,
                )
            print(f"Time taken for {model}: {time.time() - _model_time:.3f} seconds")

    else:
        print("Your bank account is safe!")

    print(f"Total time taken: {time.time() - _start_time:.3f} seconds")
