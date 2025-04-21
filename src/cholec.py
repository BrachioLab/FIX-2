import os
import io
import openai
import json
import base64
import PIL
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tfs
import datasets as hfds
from diskcache import Cache

cache = Cache(".cholec_cache")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
            tfs.Resize(image_size), # for datasets version too old, the dimension can be (3, H, W) and this will break
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

def image_to_base64(image: torch.Tensor | np.ndarray | PIL.Image.Image) -> str:
    """
    Convert an image to a base64 string.
    
    Args:
        image: Input image as torch.Tensor, np.ndarray, or PIL.Image.Image
        
    Returns:
        str: Base64 encoded PNG image string
    """
    # Convert to numpy array if tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Convert to uint8 and scale to 0-255
    image = (image * 255).astype(np.uint8)
    
    # Handle channel-first format
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # Convert to PIL Image
    pil_image = PIL.Image.fromarray(image)
    
    # Convert to base64
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
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


def get_llm_generated_answer(image) -> dict:
    """
    Get the LLM-generated answer for a given image.
    Args:
        image: A torch.Tensor, np.ndarray, or PIL.Image.Image of the surgical scene
    Returns:
        dict: Contains explanation and binary masks for safe/unsafe regions
    """
    system_prompt = """You are an expert gallbladder surgeon."""

    user_prompt = """Analyze the provided 2D image of a gallbladder surgery and:
    1. Provide a detailed explanation of your reasoning for identifying safe and unsafe regions
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
        "Explanation": "<str, reasoning chain for identifying safe and unsafe regions>",
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
                'image_url': {'url': f"data:image/png;base64,{image_to_base64(image)}"}
            },
            {'type': 'text', 'text': user_prompt}
        ]}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"} 
    )

    result = json.loads(response.choices[0].message.content)
    
    # Handle potential missing or invalid base64 strings
    for mask_name in ["SafeMask", "UnsafeMask"]:
        if mask_name in result and result[mask_name]:
            try:
                # Clean the base64 string by removing any data URL prefix
                base64_str = result[mask_name]
                if "base64," in base64_str:
                    print("Splitting base64 string")
                    base64_str = base64_str.split("base64,")[1]
                
                """
                # If the base64 string is not properly padded, add padding
                if len(base64_str) % 4 != 0:
                    padding_needed = 4 - (len(base64_str) % 4)
                    base64_str = base64_str.ljust(len(base64_str) + padding_needed, "=")
                """

                # Convert base64 to PNG bytes
                print(f"About to decode base64 string for {mask_name}, length: {len(base64_str)}")
                png_bytes = base64.b64decode(base64_str)
                print(f"PNG bytes for {mask_name}: {png_bytes}")
                
                # Convert PNG bytes to tensor
                result[mask_name] = base64_to_image(png_bytes, format="tensor")
            except Exception as e:
                print(f"Warning: Could not decode {mask_name}: {e}")
                result[mask_name] = None
        else:
            result[mask_name] = None
    
    return result, response.choices[0].message.content
