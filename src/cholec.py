# Standard library imports
import os
import io
import base64
import re
from typing import Any
# Third-party imports
import numpy as np
import torch
import PIL
import openai
from torch.utils.data import Dataset
from torchvision import transforms as tfs
import datasets as hfds
from diskcache import Cache

# Local imports
from llms import MyOpenAIModel
from prompts.claim_decomposition import decomposition_cholec
from prompts.relevance_filtering import relevance_cholec
from prompts.expert_alignment import alignment_cholec
from prompts.explanations import cholec_prompt, vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline


cache = Cache(".cholec_cache")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# default_model = "gpt-4o"
default_model = "gpt-4.1-mini"


class CholecExample:
    def __init__(
        self,
        id: str,
        image: torch.Tensor,
        organ_masks: list[torch.Tensor],
        gonogo_masks: list[torch.Tensor],
        llm_explanation: str,
    ):
        """
        Args:
            image: The image of the gallbladder surgery.
            organ_masks: The masks of the organs.
            gonogo_masks: The masks of the safe/unsafe regions.
            llm_explanation: The explanation of the safe/unsafe regions.
        """
        self.id = id
        self.image = image
        self.organ_masks = organ_masks
        self.gonogo_masks = gonogo_masks
        self.llm_explanation = llm_explanation
        self.all_claims = []
        self.relevant_claims = []
        self.alignment_scores = []
        self.alignment_categories = []
        self.alignment_reasonings = []
        self.final_alignment_score = 0

    def to_dict(self):
        return {
            "id": self.id,
            "llm_explanation": self.llm_explanation,
            "all_claims": self.all_claims,
            "relevant_claims": self.relevant_claims,
            "alignment_scores": self.alignment_scores,
            "alignment_categories": self.alignment_categories,
            "alignment_reasonings": self.alignment_reasonings,
            "final_alignment_score": self.final_alignment_score,
        }


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

    llm = MyOpenAIModel(model_name=model)

    if baseline.lower() == "vanilla":
        prompt = cholec_prompt.replace("[[BASELINE_PROMPT]]", vanilla_baseline)
    elif baseline.lower() == "cot":
        prompt = cholec_prompt.replace("[[BASELINE_PROMPT]]", cot_baseline)
    elif baseline.lower() == "socratic":
        prompt = cholec_prompt.replace("[[BASELINE_PROMPT]]", socratic_baseline)
    elif baseline.lower() == "least_to_most":
        prompt = cholec_prompt.replace("[[BASELINE_PROMPT]]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid baseline: {baseline}")

    if isinstance(image, list):
        prompts = [(prompt, i) for i in image]
        responses = llm(prompts)
        return responses

    else:
        response = llm((prompt, image))
        return response


def isolate_individual_features(
    explanation: str | list[str],
    model_name: str = default_model,
) -> list[str]:
    """
    Isolate individual features from the explanation by breaking it down into atomic claims.

    Args:
        explanation (str): The explanation text to break down into claims
        model (str): The OpenAI model to use for processing

    Returns:
        list[str]: A list of atomic claims extracted from the explanation
    """

    llm = MyOpenAIModel(model_name=model_name)

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

    prompts = [(relevance_cholec.format(claim), example_image) for claim in atomic_claims]
    llm = MyOpenAIModel(model_name=model)
    results = llm(prompts)

    relevant_claims = [
        claim for claim, result in zip(atomic_claims, results)
        if "relevance: yes" in result.lower()
    ]

    return relevant_claims


def calculate_expert_alignment_score(
    atomic_claims: list[str],
    model_name: str = default_model,
) -> dict:
    """
    Computes the individual (and overall) alignment score of all the relevant atomic claims.

    Args:
        atomic_claims (list[str]): A list of strings where each string is a relevant claim.
        model_name (str): The model to use for evaluation.

    Returns:
        dict: A dictionary containing:
            - alignment_scores: Mapping of each claim to its alignment score (1-5)
            - total_score: Overall alignment score across all claims
    """

    llm = MyOpenAIModel(model_name=model_name)
    prompts = [alignment_cholec.replace("[[CLAIM]]", claim) for claim in atomic_claims]
    responses = llm(prompts)
    return responses

