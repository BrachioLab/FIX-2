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

from typing import Callable

from prompts.explanations import massmaps_prompt, vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline
from prompts.claim_decomposition import decomposition_massmaps
from prompts.relevance_filtering import relevance_massmaps, load_relevance_massmaps_prompt
from prompts.expert_alignment import alignment_massmaps

cache = Cache(os.environ.get("CACHE_DIR"))

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

def get_llm_generated_answer(
    example: list[str] | str | torch.Tensor, #Image | Timeseries,
    method: str = "vanilla",
    model: str = "gpt-4o",
    massmap_to_pil_norm: Callable = massmap_to_pil_norm,
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the emotion classification task.
    """

    if method == 'least_to_most':
        method = 'subq'

    if method == "vanilla":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", '')
    elif method == "cot":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", cot_baseline)
    elif method == "socratic":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", socratic_baseline)
    elif method == "subq":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid method: {method}")

    prompt = prompt.replace(
        '[LAST_IMAGE_NUM]',
        '1'
    )
    

    image_pil = [massmap_to_pil_norm(example)]

    llm_response = get_llm_output(prompt, image_pil, model=model)

    response_split = [r.strip() for r in llm_response.split("\n") if r.strip() != "" \
        and r.strip().startswith("Explanation:") or r.strip().startswith("Prediction:")]
    try:
        
        explanation = response_split[0].split("Explanation: ")[1].strip()
        answer = response_split[-1].split("Prediction: ")[1].strip()
        # split the answer into Omega_m and sigma_8
        answer = answer.split(", ")
        answer = {
            answer[0].split(": ")[0]: parse_float(answer[0].split(": ")[1]), 
            answer[1].split(": ")[0]: parse_float(answer[1].split(": ")[1])
        }
        
        return answer, explanation
    except Exception as e:
        print(f"Error in parsing response {llm_response}")
        # import pdb; pdb.set_trace()
        return None, None

  

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
        all_all_claims: list[list[str]] = [
            [c.strip() for c in result.split("\n") if c.strip()]
            for result in results
        ]
        return all_all_claims
    else:
        raw_output = llm(decomposition_massmaps.format(explanation))
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

def calculate_expert_alignment_scores(
    claims: list[str],
    model: str = 'gpt-4o',
) -> list[dict]:
    """
    Computes the individual (and overall) alignment score of all the relevant claims.

    Args:
        claims (list[str]): A list of strings where each string is a relevant claim.
        model (str): The model to use for evaluation.

    Returns:
        dict: A dictionary containing:
            - alignment_scores: Mapping of each claim to its alignment score (1-5)
            - total_score: Overall alignment score across all claims
    """

    llm = load_model(model)
    prompts = [alignment_massmaps.replace("[[CLAIM]]", claim) for claim in claims]
    responses = llm(prompts)

    results = []
    for i, response in enumerate(responses):
        clean_response = [s.strip() for s in response.split("\n") if s.strip()]
        try:
            if len(clean_response) == 4:
                category = clean_response[0].split(": ")[1]
                category_id = int(clean_response[1].split(": ")[1])
                alignment = float(clean_response[2].split(": ")[1])
                reasoning = clean_response[3].split(": ")[1]

                results.append({
                    "Claim": claims[i],
                    "Category": category,
                    "Category ID": category_id,
                    "Alignment": alignment,
                    "Reasoning": reasoning,
                })

        except Exception as e:
            continue

    return results
