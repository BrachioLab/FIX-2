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


def image_to_base64(
    image: torch.Tensor | np.ndarray | PIL.Image.Image,
    include_url_prefix: bool = True,
    image_format: str = "PNG"
) -> str:
    """
    Convert an image to a base64-encoded string, optionally with a data URL prefix.
    
    This function handles various input image formats (PyTorch tensor, NumPy array, or PIL Image)
    and converts them to a standardized base64 string representation. The output can be used
    for web-based image transmission or storage.
    
    Args:
        image: Input image in one of the following formats:
            - torch.Tensor: PyTorch tensor (C,H,W) or (H,W,C)
            - np.ndarray: NumPy array (H,W,C) or (C,H,W)
            - PIL.Image.Image: PIL Image object
        include_url_prefix: If True, prepends "data:image/png;base64," to the output string
        image_format: The format to save the image in (default: "PNG")
        
    Returns:
        str: Base64-encoded image string, optionally with data URL prefix
        
    Raises:
        ValueError: If the input image format is invalid or conversion fails
    """
    try:
        # Convert to numpy array if tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure image is in uint8 format with values 0-255
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Handle channel-first format (C,H,W) -> (H,W,C)
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        # Convert to PIL Image
        pil_image = PIL.Image.fromarray(image)
        
        # Convert to base64
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=image_format)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            if include_url_prefix:
                return f"data:image/{image_format.lower()};base64,{base64_str}"
            return base64_str
            
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")


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

    messages = [
        {'role': 'system', 'content': [
            {'type': 'text', 'text': system_prompt}
        ]},
        {'role': 'user', 'content': [
            {
                "type": 'image_url',
                'image_url': {'url': image_to_base64(image)}
            },
            {'type': 'text', 'text': user_prompt}
        ]}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"} 
    )

    result = json.loads(response.choices[0].message.content)
    
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


def is_claim_relevant(
    example_image: PIL.Image.Image | torch.Tensor | np.ndarray,
    atomic_claim: str,
    model: str = default_model,
) -> bool:
    """
    Check if an atomic claim is relevant to the example.

    Args:
        example: The input image from the cholecystectomy dataset
        atomic_claim: A single claim extracted from the explanation
        model: The OpenAI model to use

    Returns:
        bool: True if the claim is relevant to the example, False otherwise
    """

    prompt = relevance_cholec.format(atomic_claim)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(example_image)}
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    result = response.choices[0].message.content
    return "relevance: yes" in result.lower()


def distill_relevant_features(
    example_image: PIL.Image.Image | torch.Tensor | np.ndarray,
    atomic_claims: list[str],
    model: str = default_model,
) -> list[str]:
    """
    Distill the relevant features from the atomic claims.
    """
    relevant_claims = []
    for claim in atomic_claims:
        if is_claim_relevant(example_image, claim, model):
            relevant_claims.append(claim)
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
