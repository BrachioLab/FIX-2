import os
import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types as genai_types
import torch
import torchvision.transforms.functional as tvtf
import numpy as np
import PIL.Image
import io
import base64


def is_image(x: Any) -> bool:
    """Check if the input is an image in a supported format."""
    return isinstance(x, (PIL.Image.Image, torch.Tensor, np.ndarray))


def image_to_base64(
    image: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
    image_format: str = "PNG"
) -> str:
    """
    Convert an image to a base64-encoded string.
    
    Args:
        image: Input image in one of the following formats:
            - torch.Tensor: PyTorch tensor (C,H,W) or (H,W,C)
            - np.ndarray: NumPy array (H,W,C) or (C,H,W)
            - PIL.Image.Image: PIL Image object
        image_format: The format to save the image in (default: "PNG")

    Returns:
        str: Base64-encoded image string

    Raises:
        ValueError: If the input image format is invalid or conversion fails
    """
    try:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        pil_image = PIL.Image.fromarray(image)
        
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=image_format)
            return base64.standard_b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")


def to_pil_image(x: Any) -> PIL.Image.Image:
    """Convert an image to a PIL Image object."""
    if isinstance(x, PIL.Image.Image):
        return x
    elif isinstance(x, torch.Tensor):
        return tvtf.to_pil_image(x)
    elif isinstance(x, np.ndarray):
        return PIL.Image.fromarray(x)
    else:
        raise ValueError(f"Invalid image type: {type(x)}")


class MyLLM(ABC):
    """Abstract base class for LLM wrappers."""

    def __call__(
        self,
        prompts: Union[str, List[Union[str, tuple]]],
        batch_size: int = 24,
        **kwargs
    ):
        """Process one or more prompts through the LLM API.
        
        Args:
            prompts: Single prompt string or list of prompts
            batch_size: Number of prompts to process in parallel
            **kwargs: Additional keyword arguments to pass to the API call
        Returns:
            Single response string or list of response strings
        """
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, **kwargs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, **kwargs) for p in prompts]
            return [f.result() for f in futures]
            
    @abstractmethod
    def one_call(self, prompt: Union[str, tuple], **kwargs):
        """Make a single API call to the LLM service."""
        raise NotImplementedError("Subclasses must implement this method")


class MyOpenAIModel(MyLLM):
    """OpenAI API wrapper implementation."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def one_call(self, prompt) -> str:
        if isinstance(prompt, str):
            content = [{"type": "text", "text": prompt}]
        elif isinstance(prompt, tuple):
            content = []
            for p in prompt:
                if isinstance(p, str):
                    content.append({"type": "text", "text": p})
                elif is_image(p):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_to_base64(p,'PNG')}"}
                    })
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        messages = [{"role": "user", "content": content}]

        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if self.verbose:
                    print(f"Error calling OpenAI's API: {e}")
                time.sleep(3)

        if self.verbose:
            print("Failed to get a valid response from OpenAI")
        return ""


class MyAnthropicModel(MyLLM):
    """Anthropic API wrapper implementation."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def one_call(self, prompt) -> str:
        if isinstance(prompt, str):
            content = [{"type": "text", "text": prompt}]
        elif isinstance(prompt, tuple):
            content = []
            for p in prompt:
                if isinstance(p, str):
                    content.append({"type": "text", "text": p})
                elif is_image(p):
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_base64(p, "PNG")
                        }
                    })
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        messages = [{"role": "user", "content": content}]

        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.content[0].text.strip()
            except Exception as e:
                if self.verbose:
                    print(f"Error calling Anthropic's API: {e}")
                time.sleep(3)

        if self.verbose:
            print("Failed to get a valid response from Anthropic")
        return ""


class MyGoogleModel(MyLLM):
    """Google API wrapper implementation."""
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)

    def one_call(self, prompt) -> str:
        if isinstance(prompt, str):
            content = [prompt]
        elif isinstance(prompt, tuple):
            content = []
            for p in prompt:
                if isinstance(p, str):
                    content.append(p)
                elif is_image(p):
                    content.append(to_pil_image(p))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=content,
                    config=genai_types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                )
                return response.text.strip()

            except Exception as e:
                if self.verbose:
                    print(f"Error calling Google's API: {e}")
                time.sleep(3)

        if self.verbose:
            print("Failed to get a valid response from Google")

        return ""
