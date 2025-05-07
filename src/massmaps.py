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
        self.expert_criteria = []
        self.alignment_reasonings = []

def convert_pil_to_base64(pil_image):
    """
    Converts a PIL image to a base64-encoded string.
    """
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def normalize_inputs(inputs, mean_center=True):
    if mean_center:
        mean = inputs.mean([-1, -2], keepdim=True)
    else:
        mean = 0
    std = inputs.std([-1, -2], keepdim=True)
    inputs_normalized = (inputs - mean) / std
    return inputs_normalized

def get_custom_colormap(colors=None):
    # Define color stops and corresponding colors
    if colors is None:
        colors = [
            (-3, "#4c1c74"),   # Blue at -3
            (0, "gray"),   # Gray at 0 (below this is void)
            (3, "yellow"),   # Green at 3 (above this is cluster)
            (20, "orange")  # Yellow at 20
        ]

    # Extract positions and colors separately
    positions, color_values = zip(*colors)

    # Normalize positions to the range [0, 1]
    positions = [(p - min(positions)) / (max(positions) - min(positions)) for p in positions]

    # Create a custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", list(zip(positions, color_values)))
    return custom_cmap


def massmap_to_pil_norm(tensor):
    """
    Converts a PyTorch tensor to a PIL image.

    Parameters:
    tensor (torch.Tensor): A tensor representing the map with shape (1, H, W)

    Returns:
    PIL.Image: An image object.
    """
    # Check if the tensor is in the range 0-1, if yes, scale to 0-255
    input_normalized = normalize_inputs(tensor, mean_center=False)
    vmin=-3
    vmax=20
    
    custom_cmap = get_custom_colormap([
            (-3, "blue"),   # Blue at -3
            (0, "gray"),   # Gray at 0 (below this is void)
            (2.9, "red"),   # Gray at 0 (below this is void)
            (3, "yellow"),   # Green at 3 (above this is cluster)
            (20, "white")  # Yellow at 20
        ])
    plt.imshow(input_normalized.cpu()[0], cmap=custom_cmap, vmin=vmin, vmax=vmax)
    # plt.imshow(tensor[0])
    plt.axis('off')  # remove axes if desired

    # Save the displayed image to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Reset buffer position
    buf.seek(0)

    # Load the image with PIL
    pil_image = PIL.Image.open(buf)
    return pil_image

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
def get_llm_output(prompt, images=None, system_prompt='', model='gpt-4o'):
    """
    prompt: str
    images: list of PIL images
    system_prompt: str
    """
    messages = get_messages(prompt, images, system_prompt)
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
    example: str | torch.Tensor, #Image | Timeseries,
    method: str = "vanilla",
    model: str = "gpt-4o"
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

    image_pil = massmap_to_pil_norm(example)
    response = get_llm_output(prompt, [image_pil], model=model)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None, None

    try:
        response_split = [r.strip() for r in response.split("\n") if r.strip() != ""]
        answer = response_split[0].split("Prediction: ")[1].strip()
        # split the answer into Omega_m and sigma_8
        answer = answer.split(", ")
        answer = {answer[0].split(": ")[0]: float(answer[0].split(": ")[1]), answer[1].split(": ")[0]: float(answer[1].split(": ")[1])}
        explanation = response_split[1].split("Explanation: ")[1].strip()
        return answer, explanation
    except:
        print(response)
        return None, None

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
    img_dir = here / "prompts"                       # e.g. .../your_script_folder/images

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
    if system_prompt is None:
        system_prompt = alignment_massmaps
        
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
    
