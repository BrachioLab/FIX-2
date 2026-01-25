import os
import concurrent.futures
import time
from typing import Any, List, Optional, Union, Tuple
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types as genai_types
import numpy as np
import torch
import torchvision.transforms.functional as tvtf
import PIL.Image
import io
import base64
import diskcache
import pickle
import hashlib
from pathlib import Path

# Create a cache in this directory
cache = diskcache.Cache(Path(__file__).parent / ".llms.py.cache")


# ==============================
# Shared image & cache utilities
# ==============================

def is_image(x: Any) -> bool:
    """Check if the input is an image in a supported format."""
    return isinstance(x, (PIL.Image.Image, torch.Tensor, np.ndarray))

def image_to_base64(
    image: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
    image_format: str = "PNG"
) -> str:
    """
    Convert an image-like (tensor/ndarray/PIL) to base64 PNG.
    Uses to_pil_image() so it’s robust to shapes/dtypes.
    """
    try:
        pil_img = to_pil_image(image)
        pil_img.load()
        with io.BytesIO() as buffer:
            pil_img.save(buffer, format=image_format)
            return base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")

def to_pil_image(x: Any) -> PIL.Image.Image:
    """
    Robust conversion:
      - supports (H, W), (C, H, W), (H, W, C)
      - auto-detects channel axis
      - if channels > 4, collapses to 1-channel by averaging
      - normalizes bool/int to float; clamps floats to [0,1]
    """
    def _finish_t(t: torch.Tensor) -> PIL.Image.Image:
        # Clamp & scale if float
        if t.is_floating_point():
            t = t.clamp(0, 1)
        return tvtf.to_pil_image(t)

    if isinstance(x, PIL.Image.Image):
        return x

    # ---- Torch tensor path ----
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        # normalize dtype
        if t.dtype == torch.bool:
            t = t.to(torch.uint8) * 255
        elif not torch.is_floating_point(t):
            t = t.to(torch.float32)

        if t.ndim == 2:
            # (H, W)
            return _finish_t(t)

        if t.ndim == 3:
            H, W = t.shape[-2], t.shape[-1]
            C_first, C_last = t.shape[0], t.shape[-1]

            # Heuristic: which axis is channel?
            # Prefer an axis with size in {1,3,4}; otherwise assume CHW if first dim != H or W
            channel_axis = None
            if C_first in (1, 3, 4):
                channel_axis = 0
            elif C_last in (1, 3, 4):
                channel_axis = 2
            else:
                # If first dim looks like channels (not H or W), treat as CHW
                channel_axis = 0 if C_first not in (H, W) else 2

            # Move channels to first dimension (CHW)
            if channel_axis == 2:
                t = t.permute(2, 0, 1)  # HWC -> CHW
            elif channel_axis == 0:
                pass                    # already CHW
            else:
                # Fallback: assume CHW
                t = t

            C = t.shape[0]
            if C not in (1, 3, 4):
                # Collapse channels >4 down to single channel
                t = t.mean(dim=0, keepdim=True)  # (1, H, W)

            return _finish_t(t)

        raise ValueError(f"Unsupported tensor ndim: {t.ndim}")

    # ---- NumPy path ----
    if isinstance(x, np.ndarray):
        a = x
        if a.dtype == np.bool_:
            a = a.astype(np.uint8) * 255
        elif not np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float32)

        if a.ndim == 2:
            # (H, W)
            return tvtf.to_pil_image(a)

        if a.ndim == 3:
            H, W = a.shape[0], a.shape[1]
            C_first, C_last = a.shape[0], a.shape[-1]

            # Detect channel axis
            if C_first in (1, 3, 4):
                chw = a
            elif C_last in (1, 3, 4):
                chw = np.transpose(a, (2, 0, 1))
            else:
                # Assume CHW if first dim not H/W; else assume HWC
                if C_first not in (a.shape[-2], a.shape[-1]):
                    chw = a
                else:
                    chw = np.transpose(a, (2, 0, 1))  # HWC -> CHW

            C = chw.shape[0]
            if C not in (1, 3, 4):
                chw = chw.mean(axis=0, keepdims=True)  # (1, H, W)

            if np.issubdtype(chw.dtype, np.floating):
                chw = np.clip(chw, 0.0, 1.0).astype(np.float32)

            return tvtf.to_pil_image(torch.from_numpy(chw))

        raise ValueError(f"Unsupported ndarray ndim: {a.ndim}")

    raise ValueError(f"Invalid image type: {type(x)}")




def get_cache_key(model_name: str, prompt: Any, system_prompt: Optional[str] = None) -> str:
    """
    Convert a (model_name, optional system_prompt, prompt) into a stable hash string.
    Backward compatible: callers not using system_prompt can omit it.
    """
    # Flatten into a list of strings (text or base64 image) for hashing
    def _serialize_one(p: Any) -> str:
        if isinstance(p, str):
            return p
        elif is_image(p):
            return image_to_base64(to_pil_image(p), "PNG")
        else:
            raise ValueError(f"Invalid prompt type for hashing: {type(p)}")

    objs: List[str] = [model_name]
    if system_prompt is not None:
        objs.append(system_prompt)

    if isinstance(prompt, str):
        objs.append(prompt)
    elif isinstance(prompt, tuple):
        for p in prompt:
            if isinstance(p, str):
                objs.append(p)
            elif is_image(p):
                objs.append(image_to_base64(p, "PNG"))
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    return hashlib.sha256(pickle.dumps(tuple(objs))).hexdigest()


# =======================
# Cloud LLMs (original)
# =======================

def load_model(model_name: str | object, api_key: Optional[str] = None, **kwargs):
    """
    Attempt to load the model based on the name. Supports:
    - OpenAI ("gpt", or model startswith "o")
    - Anthropic ("claude")
    - Google ("gemini")
    - Open-source VLMs (names containing "llava", "qwen", "pixtral")
    """
    if not isinstance(model_name, str):
        return model_name
    lname = model_name.lower()

    # VLMs first (explicit names)
    if "llava" in lname:
        return LLaVAModel(model_name=model_name, **kwargs)
    if "qwen" in lname and ("vl" in lname or "2.5" in lname):
        return QwenVLModel(model_name=model_name, **kwargs)
    if "pixtral" in lname:
        return PixtralModel(model_name=model_name, **kwargs)
    
    # Cloud text (original)
    if "gpt" in lname or model_name.startswith("o"):
        return MyOpenAIModel(model_name=model_name, api_key=api_key, **kwargs)
    elif "claude" in lname:
        return MyAnthropicModel(model_name=model_name, api_key=api_key, **kwargs)
    elif "gemini" in lname:
        return MyGoogleModel(model_name=model_name, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


class MyOpenAIModel:
    """OpenAI API wrapper implementation."""
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        max_tokens: int = 2048,
        batch_size: int = 24,
        use_cache: bool = True,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # if api_key is None:
        #     key_path = os.path.join(
        #         os.path.dirname(__file__), "..", "API_KEY.txt"
        #     )
        #     key_path = os.path.abspath(key_path)

        #     with open(key_path, "r") as f:
        #         api_key = f.read().strip()

        # self.api_key = api_key
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=self.api_key)

    def __call__(self, prompts: Union[str, List[Union[str, tuple]]]):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt) -> str:
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt))
            if ret is not None and ret != "":
                return ret

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
        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                )
                response_text = response.choices[0].message.content.strip()
                if response_text != "":
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error calling OpenAI's API: {e}")
                time.sleep(3)

        if self.use_cache and response_text != "":
            cache.set(get_cache_key(self.model_name, prompt), response_text)

        return response_text


class MyAnthropicModel:
    """Anthropic API wrapper implementation."""
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 24,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def __call__(self, prompts: Union[str, List[Union[str, tuple]]]):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt) -> str:
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt))
            if ret is not None and ret != "":
                return ret

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
        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                # note: anthropic messages.create signature may differ by SDK version
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_text = response.content[0].text.strip()
                if response_text != "":
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error calling Anthropic's API: {e}")
                time.sleep(3)

        if self.use_cache and response_text != "":
            cache.set(get_cache_key(self.model_name, prompt), response_text)

        return response_text


class MyGoogleModel:
    """Google API wrapper implementation."""
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 24,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose

        self.use_vertex = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        if self.use_vertex:
            try:
                from vertexai import init as vertex_init
                from vertexai.generative_models import GenerativeModel, Image as VertexImage
            except Exception as e:
                raise ValueError(f"Vertex AI SDK not available: {e}")

            project_id = (
                os.getenv("VERTEX_PROJECT_ID")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or "tfix-485319"
            )
            location = os.getenv("VERTEX_LOCATION") or "us-central1"
            vertex_init(project=project_id, location=location)
            self.vertex_model = GenerativeModel(self.model_name)
            self.vertex_image_cls = VertexImage
        else:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            self.client = genai.Client(api_key=self.api_key)

    def __call__(self, prompts: Union[str, List[Union[str, tuple]]]):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt) -> str:
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt))
            if ret is not None and ret != "":
                return ret

        def _to_vertex_image(pil_img: PIL.Image.Image):
            if hasattr(self.vertex_image_cls, "from_bytes"):
                with io.BytesIO() as buffer:
                    pil_img.save(buffer, format="PNG")
                    return self.vertex_image_cls.from_bytes(buffer.getvalue())
            if hasattr(self.vertex_image_cls, "from_pil_image"):
                return self.vertex_image_cls.from_pil_image(pil_img)
            return pil_img

        if isinstance(prompt, str):
            content = [prompt]
        elif isinstance(prompt, tuple):
            content = []
            for p in prompt:
                if isinstance(p, str):
                    content.append(p)
                elif is_image(p):
                    pil_img = to_pil_image(p)
                    content.append(_to_vertex_image(pil_img) if self.use_vertex else pil_img)
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                if self.use_vertex:
                    response = self.vertex_model.generate_content(content)
                else:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=content,
                        config=genai_types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        )
                    )
                response_text = response.text.strip()
                if response_text != "":
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error calling Google's API: {e}")
                time.sleep(3)

        if self.use_cache and response_text != "":
            cache.set(get_cache_key(self.model_name, prompt), response_text)

        return response_text


# =======================
# Open-source VLMs
# =======================

class LLaVAModel:
    """LLaVA 1.6 (LLaVA-NeXT) model using vLLM if available, else transformers."""
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        use_vllm: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.use_vllm = use_vllm
        self.device = device
        self._load_model()

    def _load_model(self):
        if self.use_vllm:
            try:
                import multiprocessing
                if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                    except RuntimeError:
                        self.use_vllm = False
                        return self._load_transformers_model()

                from vllm import LLM, SamplingParams
                if self.verbose:
                    print(f"Loading LLaVA with vLLM: {self.model_name}")
                self.model = LLM(
                    model=self.model_name,
                    dtype="auto",
                    max_model_len=8192,
                    trust_remote_code=True,
                    limit_mm_per_prompt={"image": 10},
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.processor = None
                return
            except Exception as e:
                if self.verbose:
                    print(f"vLLM failed for LLaVA: {e}; falling back to HF")
                self.use_vllm = False
        self._load_transformers_model()

    def _load_transformers_model(self):
        if "v1.6" in self.model_name or "next" in self.model_name.lower():
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
            model_class = LlavaNextForConditionalGeneration
            processor_class = LlavaNextProcessor
        else:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            model_class = LlavaForConditionalGeneration
            processor_class = AutoProcessor

        if self.verbose:
            print(f"Loading LLaVA with HF: {self.model_name}")

        self.model = model_class.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self.processor = processor_class.from_pretrained(self.model_name)

    def __call__(self, prompts: Union[str, Tuple], system_prompt: Optional[str] = None):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as ex:
            futures = [ex.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        cache_key = None
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            ret = cache.get(cache_key)
            if ret:
                return ret

        # Build text + collect images
        if isinstance(prompt, str):
            text_prompt = prompt
            images: List[PIL.Image.Image] = []
            ti = [("text", prompt)]
        elif isinstance(prompt, tuple):
            ti = []
            images = []
            for p in prompt:
                if isinstance(p, str):
                    ti.append(("text", p))
                elif is_image(p):
                    images.append(to_pil_image(p))
                    ti.append(("image", None))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            if self.use_vllm and images:
                parts = []
                for t, c in ti:
                    parts.append(c if t == "text" else "<image>")
                text_prompt = " ".join([x for x in parts if x])
            else:
                text_prompt = " ".join([c for t, c in ti if t == "text"])
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        # Add system prompt (simple header)
        if system_prompt:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Generate
        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                if self.use_vllm:
                    if images:
                        from vllm import LLM  # type: ignore
                        outputs = self.model.generate(
                            {
                                "prompt": full_prompt,
                                "multi_modal_data": {"image": images if len(images) > 1 else images[0]}
                            },
                            sampling_params=self.sampling_params
                        )
                    else:
                        outputs = self.model.generate({"prompt": full_prompt}, sampling_params=self.sampling_params)

                    response_text = "".join(o.outputs[0].text for o in outputs).strip()
                else:
                    # transformers path
                    is_llava_next = "v1.6" in self.model_name or "next" in self.model_name.lower()
                    if is_llava_next:
                        content = []
                        if images:
                            content.append({"type": "text", "text": full_prompt})
                            for _ in images:
                                content.append({"type": "image"})
                        else:
                            content = [{"type": "text", "text": full_prompt}]
                        conversation = [{"role": "user", "content": content}]
                        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                        if images:
                            inputs = self.processor(text=text, images=images, return_tensors="pt")
                        else:
                            inputs = self.processor(text=text, return_tensors="pt")
                    else:
                        if images:
                            inputs = self.processor(text=full_prompt, images=images, return_tensors="pt")
                        else:
                            inputs = self.processor(text=full_prompt, return_tensors="pt")

                    inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_tokens,
                            temperature=self.temperature,
                            do_sample=self.temperature > 0,
                        )
                    input_len = inputs['input_ids'].shape[1]
                    gen_ids = output_ids[:, input_len:]
                    response_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"❌ LLaVA gen error (attempt {attempt+1}/{self.num_tries_per_request}): {e}")
                time.sleep(3)

        if self.use_cache and response_text:
            cache.set(cache_key, response_text)
        return response_text


class QwenVLModel:
    """Qwen-2.5-VL model with vLLM or transformers."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 1,
        verbose: bool = True,
        use_vllm: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.use_vllm = use_vllm
        self.device = device
        self._load_model()

    def _load_model(self):
        if self.use_vllm:
            try:
                from vllm import LLM, SamplingParams
                if self.verbose:
                    print(f"Loading Qwen-VL with vLLM: {self.model_name}")
                self.model = LLM(
                    model=self.model_name,
                    dtype="auto",
                    max_model_len=8192,
                    trust_remote_code=True,
                    limit_mm_per_prompt={"image": 10},
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.processor = None
                return
            except Exception as e:
                if self.verbose:
                    print(f"vLLM failed for Qwen-VL: {e}; falling back to HF")
                self.use_vllm = False
        self._load_transformers_model()

    def _load_transformers_model(self):
        from transformers import AutoProcessor
        if self.verbose:
            print(f"Loading Qwen-VL with HF: {self.model_name}")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        except ImportError:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except Exception:
            self.process_vision_info = None
            if self.verbose:
                print("qwen_vl_utils not available; using standard processing")

    def __call__(self, prompts: Union[str, Tuple], system_prompt: Optional[str] = None):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as ex:
            futures = [ex.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        cache_key = None
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            ret = cache.get(cache_key)
            if ret:
                return ret

        # Build interleaved content
        if isinstance(prompt, str):
            text_prompt = prompt
            images = []
            structured = [("text", prompt)]
        elif isinstance(prompt, tuple):
            structured, texts, images = [], [], []
            for p in prompt:
                if isinstance(p, str):
                    texts.append(p)
                    structured.append(("text", p))
                elif is_image(p):
                    img = to_pil_image(p)
                    images.append(img)
                    structured.append(("image", img))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            text_prompt = " ".join(texts)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        # Messages per Qwen chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        has_multiple_images = len(images) > 1
        if has_multiple_images and isinstance(prompt, tuple):
            # preserve structure
            for t, c in structured:
                if t == "text" and c.strip():
                    user_content.append({"type": "text", "text": c})
                elif t == "image":
                    user_content.append({"type": "image", "image": c})
        else:
            for img in images:
                user_content.append({"type": "image", "image": img})
            if text_prompt.strip():
                user_content.append({"type": "text", "text": text_prompt})

        messages.append({"role": "user", "content": user_content})

        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                if self.use_vllm:
                    # Manually format prompt for vLLM
                    text_parts = []
                    imgs = []
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]
                        if role == "system":
                            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                        elif role == "user":
                            if isinstance(content, list):
                                user_text = ""
                                for item in content:
                                    if item.get("type") == "text":
                                        user_text += (item["text"] + " ")
                                    elif item.get("type") == "image":
                                        imgs.append(item["image"])
                                image_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * len(imgs)
                                text_parts.append(f"<|im_start|>user\n{image_tokens}{user_text.strip()}<|im_end|>")
                    text_parts.append("<|im_start|>assistant\n")
                    text = "\n".join(text_parts)

                    if imgs:
                        outputs = self.model.generate(
                            {"prompt": text, "multi_modal_data": {"image": imgs if len(imgs) > 1 else imgs[0]}},
                            sampling_params=self.sampling_params
                        )
                    else:
                        outputs = self.model.generate({"prompt": text}, sampling_params=self.sampling_params)

                    response_text = "".join(o.outputs[0].text for o in outputs).strip()
                else:
                    # transformers
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if hasattr(self, 'process_vision_info') and self.process_vision_info is not None:
                        image_inputs, video_inputs = self.process_vision_info(messages)
                        inputs = self.processor(
                            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
                        )
                    else:
                        imgs = []
                        for msg in messages:
                            if msg["role"] == "user" and isinstance(msg["content"], list):
                                for item in msg["content"]:
                                    if item.get("type") == "image":
                                        imgs.append(item["image"])
                        if imgs:
                            inputs = self.processor(text=text, images=imgs, return_tensors="pt")
                        else:
                            inputs = self.processor(text=text, return_tensors="pt")

                    inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    with torch.no_grad():
                        gen_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_tokens,
                            temperature=self.temperature,
                            do_sample=self.temperature > 0,
                        )
                    in_ids = inputs['input_ids'] if 'input_ids' in inputs else inputs.get('input_ids')
                    trimmed = [out[len(inp):] for inp, out in zip(in_ids, gen_ids)]
                    response_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"❌ Qwen-VL gen error (attempt {attempt+1}/{self.num_tries_per_request}): {e}")
                time.sleep(3)

        if self.use_cache and response_text:
            cache.set(cache_key, response_text)
        return response_text


class PixtralModel:
    """Pixtral-12B with vLLM."""
    def __init__(
        self,
        model_name: str = "mistralai/Pixtral-12B-2409",
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.device = device
        self._load_model()

    def _load_model(self):
        from vllm import LLM, SamplingParams
        if self.verbose:
            print(f"Loading Pixtral with vLLM: {self.model_name}")
        self.model = LLM(
            model=self.model_name,
            tokenizer_mode="mistral",
            dtype="auto",
            max_model_len=16384,
            limit_mm_per_prompt={"image": 10},
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def __call__(self, prompts: Union[str, Tuple], system_prompt: Optional[str] = None):
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as ex:
            futures = [ex.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]

    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        cache_key = None
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            ret = cache.get(cache_key)
            if ret:
                return ret

        if isinstance(prompt, str):
            text_prompt = prompt
            images = []
            structured = [("text", prompt)]
        elif isinstance(prompt, tuple):
            structured, texts, images = [], [], []
            for p in prompt:
                if isinstance(p, str):
                    texts.append(p)
                    structured.append(("text", p))
                elif is_image(p):
                    img = to_pil_image(p)
                    images.append(img)
                    structured.append(("image", img))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            text_prompt = " ".join(texts)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        has_multi = len(images) > 1
        if has_multi and isinstance(prompt, tuple):
            for t, c in structured:
                if t == "text" and c.strip():
                    user_content.append({"type": "text", "text": c})
                elif t == "image":
                    img_b64 = image_to_base64(c)
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        else:
            for img in images:
                img_b64 = image_to_base64(img)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            if text_prompt.strip():
                user_content.append({"type": "text", "text": text_prompt})

        messages.append({"role": "user", "content": user_content})

        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                outputs = self.model.chat(messages, sampling_params=self.sampling_params)
                response_text = outputs[0].outputs[0].text.strip()
                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"❌ Pixtral gen error (attempt {attempt+1}/{self.num_tries_per_request}): {e}")
                time.sleep(3)

        if self.use_cache and response_text:
            cache.set(cache_key, response_text)
        return response_text
