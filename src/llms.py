import os
import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI
import anthropic
import torch
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
            base64_str = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
            return base64_str

    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")


class MyOpenAIModel:
    """
    A wrapper class for OpenAI's chat completion API that supports both single and batch processing.

    Usage:
        >>> llm = MyOpenAIModel("gpt-4o-mini")
        >>> image = PIL.Image.open("path/to/image.png")
        >>> out = llm([
                "What's your name?",
                ("Tell me what you think about this image", image)
            ])
        >>> print(out)
        [
            "I'm called Assistant. How can I help you today?",
            "The image is a beautiful landscape."
        ]
    """
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        batch_size: int = 24,
        verbose: bool = False
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.client = OpenAI(api_key=self.api_key)
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.batch_size = batch_size
        self.verbose = verbose

    def __call__(
        self,
        prompts: Union[str, List[Union[str, tuple]]],
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """
        Process one or more prompts through the OpenAI API.
        
        Args:
            prompts: Single prompt string or list of prompts
            response_format: Optional format specification for the response
            temperature: Controls randomness in the response (0.0 to 1.0)
            seed: Optional seed for reproducibility
            batch_size: Number of prompts to process in parallel
            
        Returns:
            Single response string or list of response strings
        """
        # Convert single prompt to list for uniform processing
        is_single_prompt = isinstance(prompts, (str, tuple))
        prompts = [prompts] if is_single_prompt else prompts
        
        # Concurrently process prompts
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [
                executor.submit(
                    self.single_call,
                    messages=self.prompt_to_messages(p),
                    response_format=response_format,
                    seed=seed,
                )
                for p in prompts
            ]

            all_responses = [f.result() for f in futures]
        
        return all_responses[0] if is_single_prompt else all_responses

    def prompt_to_messages(self, prompt: Union[str, tuple]) -> List[Dict[str, Any]]:
        """Convert a prompt to the format expected by OpenAI's API."""
        if isinstance(prompt, str):
            return [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]
    
        elif isinstance(prompt, tuple):
            content = []
            for p in prompt:
                if isinstance(p, str):
                    content.append({
                        "type": "text",
                        "text": p
                    })
                elif is_image(p):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_to_base64(p,'PNG')}"}
                    })
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")

            return [{"role": "user", "content": content}]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def single_call(self, messages, response_format=None, seed=None) -> str:
        """Make a single API call to OpenAI."""
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    seed=seed,
                    response_format=response_format,
                )

                content = response.choices[0].message.content
                if isinstance(content, str):
                    return content.strip()

                else:
                    raise ValueError(f"Invalid response content: {content}")

            except Exception as e:
                if self.verbose:
                    print(f"Error calling OpenAI: {e}")

                time.sleep(3)

        if self.verbose:
            print("Failed to get a valid response from OpenAI")

        return ""


class MyAnthropicModel:
    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        batch_size: int = 24,
        verbose: bool = False,
    ):
        """Initialize the Anthropic API wrapper.
    
        Args:
            model_name: Name of the Claude model to use
            num_tries_per_request: Number of retry attempts per API call
            verbose: Whether to print debug information
        """
        self.model_name = model_name
        
        # Try to load API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.verbose = verbose

    def __call__(self, prompts: Union[str, List[Union[str, tuple]]]) -> Union[str, List[str]]:
        """Call Anthropic API with one or more prompts.
        
        Args:
            prompts: Single prompt string or list of prompt strings
            temperature: Sampling temperature (0-1)
            batch_size: Maximum number of concurrent API calls
            
        Returns:
            Single response string or list of response strings
        """
        is_single_prompt = isinstance(prompts, (str, tuple))
        prompts = [prompts] if is_single_prompt else prompts

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [
                executor.submit(self.single_call, messages=self.prompt_to_messages(p))
                for p in prompts
            ]

            all_responses = [f.result() for f in futures]
        
        return all_responses[0] if is_single_prompt else all_responses

    def prompt_to_messages(self, prompt: Union[str, tuple]) -> List[Dict[str, Any]]:
        """Convert a prompt to the format expected by Anthropic's API."""
        if isinstance(prompt, str):
            return [{
                "role": "user",
                "content": prompt
            }]
        
        elif isinstance(prompt, tuple):
            content = []
            for p in prompt:
                if isinstance(p, str):
                    content.append({
                        "type": "text",
                        "text": p
                    })
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

            return [{"role": "user", "content": content}]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def single_call(self, messages: List[Dict[str, Any]]) -> str:
        """Make a single API call to Anthropic."""
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                content = response.content[0].text
                if isinstance(content, str):
                    return content.strip()
                else:
                    raise ValueError(f"Invalid response content: {content}")

            except Exception as e:
                if self.verbose:
                    print(f"Error calling Anthropic: {e}")

                time.sleep(3)

        if self.verbose:
            print("Failed to get a valid response from Anthropic")

        return ""
