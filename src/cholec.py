# Standard library imports
import os
import io
import json
import base64
import re
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


cache = Cache(".cholec_cache")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

default_model = "gpt-4o"

class CholecDataset(Dataset):
    """
    The cholecystectomy (gallbladder surgery) dataset, loaded from HuggingFace.
    The task is to find the safe/unsafe (gonogo) regions.
    The expert-specified features are the organ labels.

    For more details, see: https://huggingface.co/datasets/BrachioLab/cholec
    """

    gonogo_names: str = [
        "Background",
        "Safe",
        "Unsafe"
    ]

    organ_names: str = [
        "Background",
        "Liver",
        "Gallbladder",
        "Hepatocystic Triangle"
    ]

    def __init__(
        self,
        split: str = "train",
        hf_data_repo: str = "BrachioLab/cholec",
        image_size: tuple[int] = (360, 640)
    ):
        r"""
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
            "image": image,     # (3,H,W)
            "gonogo": gonogo,   # (H,W)
            "organs": organs,   # (H,W)
        }


def base64_to_image(base64_str: str, format: str = "tensor") -> torch.Tensor | PIL.Image.Image:
    """
    Convert a base64 string to an image.
    
    Args:
        base64_str: Base64 encoded image string
        format: Format to convert to ("tensor" or "pil")
        
    Returns:
        torch.Tensor | PIL.Image.Image: The image in the requested format
    """
    try:
        # Remove data URL prefix if present
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]
            
        # First convert base64 to bytes
        # Add padding if needed
        padding = len(base64_str) % 4
        if padding:
            base64_str += '=' * (4 - padding)
        image_bytes = base64.b64decode(base64_str)
        
        # Create PIL Image from bytes
        pil_image = PIL.Image.open(io.BytesIO(image_bytes))
        
        if format.lower() == "pil":
            return pil_image
        elif format.lower() == "tensor":
            # Convert PIL Image to tensor
            return torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        else:
            raise ValueError(f"Invalid format: {format}")
    except Exception as e:
        raise ValueError(f"Failed to convert base64 to image: {str(e)}")


def get_llm_generated_answer(
    image: torch.Tensor | np.ndarray | PIL.Image.Image,
    model: str = default_model,
    decode_base64_masks: bool = False
) -> dict[str, str | torch.Tensor | None]:
    """
    Generate a detailed surgical analysis and segmentation masks using an LLM.
    
    This function sends a surgical image to an LLM and receives back:
    1. A detailed explanation of safe/unsafe regions
    2. Binary masks for safe/unsafe regions
    
    Args:
        image: Input surgical image in tensor, numpy array, or PIL Image format
        model: Name of the LLM model to use (default: "gpt-4o")
        decode_base64_masks: If True, converts base64 mask strings to PyTorch tensors
        
    Returns:
        Dictionary containing:
            - "Explanation": Detailed text analysis of safe/unsafe regions
            - "SafeMask": Base64-encoded PNG or tensor of safe regions
            - "UnsafeMask": Base64-encoded PNG or tensor of unsafe regions
    """
    system_prompt = """You are an expert gallbladder surgeon with extensive experience in laparoscopic cholecystectomy. 
    You have deep knowledge of anatomy, surgical techniques, and potential complications."""

    user_prompt = """Analyze the provided 2D image of a gallbladder surgery and provide an extremely detailed analysis:

    1. Provide an exhaustive explanation of your reasoning for identifying safe and unsafe regions, including:
       - Detailed anatomical landmarks and their significance
       - Specific tissue types and their surgical implications
       - Potential risks and complications in unsafe regions
       - Surgical instrument considerations for each region
       - Any visible pathology or abnormalities
       - Critical structures that must be preserved
       - Step-by-step reasoning for each region's classification

    2. Generate two binary masks as grayscale PNG images (8-bit per pixel):
       - A mask showing safe regions (where surgical instruments can safely operate)
       - A mask showing unsafe regions (where surgical instruments should avoid)

    For the masks:
    - Use 0 (black) for background/unsafe regions
    - Use 255 (white) for safe regions
    - Save as 8-bit grayscale PNG format
    - Return only the base64-encoded PNG data in the image_url field
    - Do not include any data URL prefix
    - IMPORTANT: You MUST provide both masks as valid base64-encoded PNG images
    - The masks are required for the analysis to be complete
    - If you cannot generate a mask, explain why in the Explanation field
    - Each mask should be a binary image with only 0 and 255 values
    - The masks should cover the entire image area
    - Make sure the base64 encoding is complete and properly formatted

    Output format:
    ```json
    {
        "Explanation": "<str, extremely detailed reasoning chain for identifying safe and unsafe regions>",
        "SafeMask": "<str, raw base64-encoded PNG image of safe regions>",
        "UnsafeMask": "<str, raw base64-encoded PNG image of unsafe regions>"
    }
    ```
    """

    llm = MyOpenAIModel(model_name=model)
    response = llm((system_prompt, user_prompt, image), response_format={"type": "json_object"})
    result = json.loads(response)

    if decode_base64_masks:
        for mask_name in ["SafeMask", "UnsafeMask"]:
            if mask_name in result and result[mask_name]:
                try:
                    base64_str = result[mask_name]
                    result[mask_name] = base64_to_image(base64_str, format="tensor")
                except Exception as e:
                    result[mask_name] = None
            else:
                result[mask_name] = None
    
    return result


def isolate_individual_features(
    explanation: str,
    model_name: str = default_model,
) -> list[str]:
    """
    Isolate individual features from the explanation by breaking it down into atomic claims.

    Args:
        explanation (str): The explanation text to break down into claims
        model (str): The OpenAI model to use for processing

    Returns:
        list[str]: A list of atomic claims extracted from the explanation

    Raises:
        ValueError: If the model output cannot be parsed as valid JSON
    """

    llm = MyOpenAIModel(model_name=model_name)
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
    responses = llm(prompts, response_format={"type": "json_object"})

    try:
        results = [json.loads(response) for response in responses]
        return results
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")
