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


from prompts.explanations import massmaps_prompt, vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline
from prompts.claim_decomposition import decomposition_massmaps
from prompts.relevance_filtering import relevance_massmaps
from prompts.expert_alignment import alignment_massmaps

cache = Cache(os.environ.get("CACHE_DIR"))

openai.api_key = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI(api_key=openai.api_key)

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
    colors: list = None
) -> Image.Image:
    """
    Convert a (1,H,W) tensor → PIL Image (H×W), with:
      • optional mean‐centering
      • divide‐by‐std normalization
      • clip to [vmin,vmax], then min–max to [0,1]
      • apply custom colormap ⇒ RGB
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
    return Image.fromarray(rgb)

def get_messages(prompt, images=None, system_prompt=None):
    system_message = [
                {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
            ]

    image_payload = []
    if images:
        image_payload = [
            {
                "type": 'image_url',
                'image_url': {'url': f"data:image/jpeg;base64,{convert_pil_to_base64(image)}"}
            }
            for image in images
        ]

    new_message = [
        {'role': 'user', 'content': image_payload + [
            {'type': 'text', 'text': prompt}
        ]
        }
    ]
    
    messages = system_message + new_message
    
    return messages

def get_system_message(system_prompt):
    system_message = [
                {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
            ]
    return system_message

def get_example_message(image, user_text, user_prompt, assistant_text=None, assistant_prompt=None):
    if image is not None:
        image_payload = [
            {
                "type": 'image_url',
                'image_url': {'url': f"data:image/jpeg;base64,{convert_pil_to_base64(image)}"}
            }
        ]
    else:
        image_payload = []
    
    if assistant_prompt is not None and assistant_text is not None:
        if isinstance(assistant_text, (list, tuple)) or (hasattr(assistant_text, '__iter__') and not isinstance(assistant_text, str)):
            assistant_message = {'role': 'assistant', 'content': [{'type': 'text', 'text': assistant_prompt.format(*assistant_text)}]}
        else:
            assistant_message = {'role': 'assistant', 'content': [{'type': 'text', 'text': assistant_prompt.format(assistant_text)}]}
    else:
        assistant_message = None
    return user_message, assistant_message

def get_few_shot_user_assistant_messages(images_list, user_text_list, user_prompt, assistant_text_list, assistant_prompt):
    all_messages = []
    for image, user_text, assistant_text in zip(images_list, user_text_list, assistant_text_list):
        user_message, assistant_message = get_example_message(image, user_text, user_prompt, assistant_text, assistant_prompt)
        all_messages.append(user_message)
        all_messages.append(assistant_message)
    return all_messages

def text2json(text):
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        # Escape single backslashes
        json_str = json_str.replace('\\', '\\\\')
        data = json.loads(json_str)
    else:
        data = None
    return data

@cache.memoize()
def get_llm_output(prompt, images=None, model='gpt-4o'):
    """
    prompt: str
    images: list of PIL images
    system_prompt: str
    """

    llm = load_model(model)

    return llm((prompt, images))

    # messages = get_messages(prompt, images, system_prompt)
    # for i in range(3):
    #     try:
    #         response = client.chat.completions.create(
    #                 model=model,
    #                 messages=messages,
    #                 response_format={'type': 'text'},
    #                 temperature=0,
    #                 max_tokens=500,
    #                 top_p=1,
    #                 frequency_penalty=0,
    #                 presence_penalty=0
    #             )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         print("Try {}; Error: {}".format(str(i), str(e)))     
    #         time.sleep(3)
    # return "ERROR"

@cache.memoize()
def get_llm_output_from_messages(messages, model='gpt-4o'):
    """
    messages: list of messages
    """
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={'type': 'text'},
                temperature=0,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Try {}; Error: {}".format(str(i), str(e)))     
            time.sleep(3)
    return "ERROR"

def get_llm_generated_answer(
    example: list[str] | str | torch.Tensor, #Image | Timeseries,
    method: str = "vanilla",
    model: str = "gpt-4o",
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the emotion classification task.
    """


    if method == "vanilla":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", '')
    elif method == "cot":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", cot_baseline)
    elif method == "socratic":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", socratic_baseline)
    elif method == "least_to_most":
        prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid method: {method}")

    prompt = prompt.replace(
        '[LAST_IMAGE_NUM]',
        '1'
    )
    

    # llm_inputs = [(prompt, massmap_to_pil_norm(e)) for e in example]

    # llm_responses = llm(llm_inputs)

    image_pil = [massmap_to_pil_norm(example)]

    llm_response = get_llm_output(prompt, image_pil, model=model)

    response_split = [r.strip() for r in llm_response.split("\n") if r.strip() != ""]
    try:
        answer = response_split[1].split("Prediction: ")[1].strip()
        # split the answer into Omega_m and sigma_8
        answer = answer.split(", ")
        answer = {
            answer[0].split(": ")[0]: float(answer[0].split(": ")[1]), 
            answer[1].split(": ")[0]: float(answer[1].split(": ")[1])
        }
        explanation = response_split[0].split("Explanation: ")[1].strip()
        
        return answer, explanation
    except Exception as e:
        print(f"Error in parsing response {llm_response}")
        return None, None

    # images_pil = []
    # images_pil.append(massmap_to_pil_norm(example))
    
    # if method == "vanilla":
    #     prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", '')
    # elif method == "cot":
    #     prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", cot_baseline)
    # elif method == "socratic":
    #     prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", socratic_baseline)
    # elif method == "least_to_most":
    #     prompt = massmaps_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline)
    # else:
    #     raise ValueError(f"Invalid method: {method}")
        
    # prompt = prompt.replace(
    #     '[LAST_IMAGE_NUM]',
    #     str(len(images_pil))
    # )

    # response = get_llm_output(prompt, images_pil, model=model)
    # if response == "ERROR":
    #     print("Error in querying OpenAI API")
    #     return None, None

    # try:
    #     response_split = [r.strip() for r in response.split("\n") if r.strip() != ""]
    #     answer = response_split[1].split("Prediction: ")[1].strip()
    #     # split the answer into Omega_m and sigma_8
    #     answer = answer.split(", ")
    #     answer = {
    #         answer[0].split(": ")[0]: float(answer[0].split(": ")[1]), 
    #         answer[1].split(": ")[0]: float(answer[1].split(": ")[1])
    #     }
    #     explanation = response_split[0].split("Explanation: ")[1].strip()
    #     return answer, explanation
    # except:
    #     print(response)
    #     return None, None

def isolate_individual_features(
    explanation: str,
    model: str = "gpt-4o"
):
    """
    Args:
        explanation (str): The LLM-generated reasoning chain of why it gave a specific answer to an example.
        
    Returns:
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    """

    prompt = decomposition_massmaps.format(explanation)
    response = get_llm_output(prompt, model=model)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")

    return claims

def is_claim_relevant(
    example: str | torch.Tensor,
    answer: str,
    atomic_claim: str,
    threshold: float = 0.9,
    model: str = "gpt-4o"
) -> bool:
    """
    For a claim to be relevant, it must be:
        (1) Supported by the example.
        (2) Answers the question of why the LLM gave the answer it did for this specific example.

    Args:
        example (str | Image | timeseries): The input example from a dataset from which to distill the relevant features from.
        answer (str): The LLM-generated answer to the example.
        atomic_claim (str): A claim to check if it is relevant to the example.
    """

    # Get the images
    # 1. Locate the *directory that this .py file lives in*
    here = Path(__file__).resolve().parent          # .../your_script_folder

    # 2. Point to the images folder *relative to* that location
    img_dir = here / "prompts" / "data"                      # e.g. .../your_script_folder/images

    # 3. Collect every PNG/JPG (adjust the glob pattern as needed)
    few_shot_image_paths = sorted(img_dir.glob("massmaps_relevance*.png")) + \
                sorted(img_dir.glob("massmaps_relevance*.jpg"))

    # 4. Load them (returns a list of PIL Images here)
    few_shot_images = [PIL.Image.open(p) for p in few_shot_image_paths]

    assert len(few_shot_images) > 0

    prompt = relevance_massmaps.format(f"Omega_m = {answer['Omega_m']}, sigma_8 = {answer['sigma_8']}", atomic_claim)

    current_image_pil = massmap_to_pil_norm(example)

    image_payloads = []
    for image in few_shot_images:
        image_payloads.append(
            {
                "type": 'image_url',
                'image_url': {'url': f"data:image/jpeg;base64,{convert_pil_to_base64(image)}"}
            }
    )

    image_payloads.append(
            {
                "type": 'image_url',
                'image_url': {'url': f"data:image/jpeg;base64,{convert_pil_to_base64(current_image_pil)}"}
            }
        )

    user_message = {
        'role': 'user',
        'content': image_payloads + [
            {'type': 'text', 'text': prompt}
        ]
    }

    messages = [user_message]

    response = get_llm_output_from_messages(messages, model=model)

    response = response.replace("Relevance:", "").strip()
    response = response.split("\n")
    relevance = response[0].strip()
    reasoning = response[1].replace("Reasoning:", "").strip()
    return relevance, reasoning

def distill_relevant_features(
    example: str | torch.Tensor,
    answer: str,
    raw_atomic_claims: list[str],
    threshold: float = 0.9,
    model: str = "gpt-4o"
):
    """
    Args:
        example (str | Image | timeseries): The input example from a dataset from which to distill the relevant features from.
        answer (str): The LLM-generated answer to the example.
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    Returns:
        atomic_claims (list[str]): A list of strings where each string is a relevant claim.
    """
    atomic_claims = []
    for raw_atomic_claim in raw_atomic_claims:
        if is_claim_relevant(example, answer, raw_atomic_claim, model=model):
            atomic_claims.append(raw_atomic_claim)
    return atomic_claims

def calculate_expert_alignment_score(
    example_input: torch.Tensor, 
    llm_prediction: str, claim: str,
    system_prompt=None,
    model: str = "gpt-4o"
):
        
    prompt = alignment_massmaps.format(claim)
    response = get_llm_output(prompt, model=model)
    response_old = response
    # print(response)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Category:", "").strip()
    response = response.split("\n")
    response = [r for r in response if r.strip() != ""]
    category = response[0].strip()
    category_id = response[1].replace("Category ID:", "").strip()
    alignment_score = response[2].replace("Category Alignment Rating:", "").strip()
    try:
        alignment_score = float(alignment_score)
    except:
        print("ERROR: Could not convert alignment score to float")
        print(response)
        import pdb; pdb.set_trace()
        alignment_score = 0.0
    reasoning = response[3].replace("Reasoning:", "").strip()
    return category, category_id, alignment_score, reasoning
    
