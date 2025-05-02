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

from prompts.claim_decomposition import decomposition_massmaps
from prompts.relevance_filtering import relevance_massmaps

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

def massmap_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image.

    Parameters:
    tensor (torch.Tensor): A tensor representing the map with shape (1, H, W)

    Returns:
    PIL.Image: An image object.
    """
    # Check if the tensor is in the range 0-1, if yes, scale to 0-255
    plt.imshow(tensor[0])
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

@cache.memoize()
def get_llm_score(prompt, images=None, system_prompt=None, model='gpt-4o') -> Tuple[str, float]:
    if system_prompt is None:
        system_prompt = "Answer only as a YES or NO."
    messages = get_messages(prompt, images, system_prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "text"},
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logit_bias={31958: 100, 14695: 100},
        logprobs=True,
    )
    completion = response.choices[0].logprobs.content[0].token.strip().lower()
    logprob = response.choices[0].logprobs.content[0].logprob
    sleep_duration = random.uniform(0.5, 2)
    time.sleep(sleep_duration)
    return completion, math.exp(logprob)




class Image: pass

class Timeseries: pass

class AlignmentScores: pass


import json
import re


def get_llm_generated_answer(
    example: str | torch.Tensor, #Image | Timeseries,
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the emotion classification task.
    """
    prompt = """Analyze this weak lensing map data in the image provided.
This data represents cosmological observations, where each value represents the spatial distribution of matter density in the universe. 

Here is the colormap used to create the visualization of this weak lensing map:
custom_cmap = get_custom_colormap([
            (-3, "blue"),   # Blue at -3 std
            (0, "gray"),   # Gray at 0 (below this is void)
            (2.9, "red"),   # Red at 2.9 std (this is the upperbound for not being a cluster)
            (3, "yellow"),   # Yellow at 3 std (above this is cluster)
            (20, "white")  # White at 20 std
        ])
        
Predict the values for Omega_m and sigma_8 based on the information from this weak lensing map data. Provide a reasoning chain for what interpretable weak lensing map cosmological features (e.g. voids and clusters) you see from this data that you use to make such predictions. When you provide the reasoning chain, for each claim, please be specific for the actual observations you see in this particular weak lensing map. Provide a short paragraph that is around 100-200 words, and then make the predictions.

Output format:
```json
{
    "Explanation": "<str, reasoning chain for predicting Omega_m and sigma_8>",
    "Answer": {"Omega_m": <float, prediction for Omega_m>, "sigma_8": <float, prediction for sigma_8>}
}
```
 """
    
#     prompt = """This is an image of a weak lensing map. Please predict Omega_m and sigma_8 values from this and provide a reasoning chain for what interpretable cosmological features you see from this map that you use to make such predictions. Provide a short paragraph that is around 100-200 words. And then make the prediction.

# Output format:
# ```json
# {
#     "Explanation": "<str, reasoning chain for predicting Omega_m and sigma_8>",
#     "Answer": {"Omega_m": <float, prediction for Omega_m>, "sigma_8": <float, prediction for sigma_8>}
# }
# ```
# """
    system_prompt = "You are an expert cosmologist."
    
    image_pil = massmap_to_pil_norm(example)
    results = text2json(get_llm_output(prompt, [image_pil], system_prompt))

    # few_shot_images = [massmap_to_pil_norm(example['X'][0]) for example in few_shot_examples]
    # few_shot_messages = get_few_shot_user_assistant_messages(
    #     few_shot_images,
    #     user_text_list=[example['claim'] for example in few_shot_examples],
    #     user_prompt=example_prompt,
    #     assistant_text_list=[(example['relevance_answer'], example['relevance_explanation']) for example in few_shot_examples],
    #     assistant_prompt=example_assistant_prompt
    # )

    try:
        return results['Answer'], results['Explanation']
    except:
        return None, None

def isolate_individual_features(
    explanation: str
):
    """
    Args:
        explanation (str): The LLM-generated reasoning chain of why it gave a specific answer to an example.
        
    Returns:
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    """

    prompt = decomposition_massmaps.format(explanation)
    response = get_llm_output(prompt)
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
    threshold: float = 0.9
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

    # # ── optional: quick sanity check ─────────────────────────────────────────
    # print(f"Found {len(few_shot_images)} few_shot_images:")
    # for p in few_shot_image_paths:
    #     print("  •", p.name)


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

    response = get_llm_output_from_messages(messages)

    response = response.replace("Relevance:", "").strip()
    response = response.split("\n")
    relevance = response[0].strip()
    reasoning = response[1].replace("Reasoning:", "").strip()
    return relevance, reasoning

def distill_relevant_features(
    example: str | torch.Tensor,
    answer: str,
    raw_atomic_claims: list[str],
    threshold: float = 0.9
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
        if is_claim_relevant(example, answer, raw_atomic_claim):
            atomic_claims.append(raw_atomic_claim)
    return atomic_claims

def calculate_expert_alignment_score(
    example_input: torch.Tensor, 
    llm_prediction: str, claim: str,
    system_prompt=None):
    if system_prompt is None:
        system_prompt = """You are an expert cosmologist. Your task is to evaluate how well the following claims aligns with known ground truth criteria used in predicting cosmological parameters from weak lensing maps.

The ground truth criteria below represent core observational patterns that support the prediction of cosmological parameters Omega_m and sigma_8. These patterns often appear in groups of pixels in weak lensing maps:
1. **Voids:** Voids are large regions under-dense relative to the mean density (pixel intensity < 0) and appear as dark regions in the mass maps.
2. **Clusters:** Clusters are areas of concentrated high density (pixel intensity > 3std) and appear as bright dots.
3. **Super-clusters:** "Super-clusters" (containing multiple clusters) may play a special role in weak lensing maps that deserves further investigation.
4. **Spatial distribution:** The spatial distribution of matter density.

For each claim, assess how well it semantically and factually aligns with the ground truth indicators above. Avoid focusing on superficial keyword matches and evaluate the actual meaning and evidentiary alignment.

Use the following relevance scale from 1 to 5:
1: Completely contradicts: The claim fundamentally misrepresents or contradicts the criteria used in cosmological parameter prediction.
2: Mostly contradicts: The claim is largely inconsistent with known indicators or suggests irrelevant patterns.
3: Partially aligns: The claim is related but lacks a clear or accurate connection to ground truth patterns.
4: Mostly aligns: The claim captures a valid prediction cue, though with minor vagueness or lack of specificity.
5: Completely aligns: The claim is fully consistent with one or more ground truth indicators and describes meaningful observational patterns useful for prediction.
Also provide a brief justification for each score, explaining the reasoning in terms of the observed patterns and their relevance to prediction.

Input format:
Claim:
<claim 1>

Output format:
Scores:
```json
{
    "claim": "<claim 1>", 
    "score": <alignment score ranging from 1 to 5>, 
    "category": "<verbatim copy of the title of expert knowledge used (Voids/Clusters/Super-clusters/Spatial Distributions/Noise and Artifacts/Not Aligned)>",
    "explanation": "<a brief one/two sentence justification for making this decision>"}
```
"""

    alignment_prompt = """Claim:
{}
"""
    
    prompt = alignment_prompt.format(claim)
    try:
        response = get_llm_output(prompt, system_prompt=system_prompt)
        alignment_result = text2json(response)
    except:
        print("Error in querying OpenAI API")
        import pdb; pdb.set_trace()
        return None
    
    alignment_score = alignment_result['score']
    category = alignment_result['category']
    reasoning = alignment_result['explanation']
    return category, alignment_score, reasoning

if __name__ == "__main__":
    # Uncomment line below to install exlib
    # !pip install exlib
    import sys; sys.path.insert(0, "../exlib/src")
    import exlib

    import torch
    from datasets import load_dataset
    from exlib.datasets.mass_maps import MassMapsDataset

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load data
    val_dataset = MassMapsDataset(split="validation")

    X, y = val_dataset[0:2]['input'], val_dataset[0:2]['label']

    di = 0
    image = X[di]

    llm_generated_answer = get_llm_generated_answer(image)
    # 'In a weak lensing map, the distribution of dark matter can be inferred from the distortion of background galaxies due to gravitational lensing. The presence of structures, such as clusters and filaments, indicates the underlying matter distribution. \n\nFrom this map, we can estimate the values of \\(\\Omega_m\\) (the matter density parameter) and \\(\\sigma_8\\) (the amplitude of density fluctuations). A higher density of dark matter structures suggests a larger \\(\\Omega_m\\), while the degree of clustering informs us about \\(\\sigma_8\\). \n\nIf the map shows significant clustering and pronounced features, we might predict \\(\\Omega_m\\) to be around 0.3 to 0.4, indicating a substantial amount of matter in the universe. For \\(\\sigma_8\\), if the structures appear tightly clustered, values around 0.8 to 0.9 could be inferred, reflecting a high amplitude of fluctuations. Conversely, a more diffuse distribution would suggest lower values. Thus, analyzing the density and distribution of features in the map allows us to make these cosmological predictions.'

    raw_atomic_claims = isolate_individual_features(llm_generated_answer)
    # ['In a weak lensing map, the distribution of dark matter can be inferred from the distortion of background galaxies due to gravitational lensing.',
    # 'The presence of structures, such as clusters and filaments, indicates the underlying matter distribution.',
    # 'From the weak lensing map, we can estimate the values of \\(\\Omega_m\\) (the matter density parameter) and \\(\\sigma_8\\) (the amplitude of density fluctuations).',
    # 'A higher density of dark matter structures suggests a larger \\(\\Omega_m\\).',
    # 'The degree of clustering in the weak lensing map informs us about \\(\\sigma_8\\).',
    # 'If the map shows significant clustering and pronounced features, we might predict \\(\\Omega_m\\) to be around 0.3 to 0.4.',
    # 'A prediction of \\(\\Omega_m\\) around 0.3 to 0.4 indicates a substantial amount of matter in the universe.',
    # 'If the structures in the map appear tightly clustered, values of \\(\\sigma_8\\) around 0.8 to 0.9 could be inferred.',
    # 'A high amplitude of fluctuations is reflected by values of \\(\\sigma_8\\) around 0.8 to 0.9.',
    # 'A more diffuse distribution of structures in the map would suggest lower values of \\(\\sigma_8\\).',
    # 'Analyzing the density and distribution of features in the weak lensing map allows us to make cosmological predictions.']

    atomic_claims = distill_relevant_features(image, llm_generated_answer, raw_atomic_claims)
    # ['In a weak lensing map, the distribution of dark matter can be inferred from the distortion of background galaxies due to gravitational lensing.',
    # 'The presence of structures, such as clusters and filaments, indicates the underlying matter distribution.',
    # 'From the weak lensing map, we can estimate the values of \\(\\Omega_m\\) (the matter density parameter) and \\(\\sigma_8\\) (the amplitude of density fluctuations).',
    # 'A higher density of dark matter structures suggests a larger \\(\\Omega_m\\).',
    # 'The degree of clustering in the weak lensing map informs us about \\(\\sigma_8\\).',
    # 'A more diffuse distribution of structures in the map would suggest lower values of \\(\\sigma_8\\).',
    # 'Analyzing the density and distribution of features in the weak lensing map allows us to make cosmological predictions.']

    alignment_scores = calculate_expert_alignment_score(atomic_claims)
    # {'alignment_scores': [{'claim': 'In a weak lensing map, the distribution of dark matter can be inferred from the distortion of background galaxies due to gravitational lensing.',
    # 'score': 5},
    # {'claim': 'The presence of structures, such as clusters and filaments, indicates the underlying matter distribution.',
    # 'score': 5},
    # {'claim': 'From the weak lensing map, we can estimate the values of \\\\(\\\\Omega_m\\\\) (the matter density parameter) and \\\\(\\\\sigma_8\\\\) (the amplitude of density fluctuations).',
    # 'score': 5},
    # {'claim': 'A higher density of dark matter structures suggests a larger \\\\(\\\\Omega_m\\\\).',
    # 'score': 5},
    # {'claim': 'The degree of clustering in the weak lensing map informs us about \\\\(\\\\sigma_8\\\\).',
    # 'score': 5},
    # {'claim': 'A more diffuse distribution of structures in the map would suggest lower values of \\\\(\\\\sigma_8\\\\).',
    # 'score': 5},
    # {'claim': 'Analyzing the density and distribution of features in the weak lensing map allows us to make cosmological predictions.',
    # 'score': 5}],
    # 'total_score': 5}
    print(alignment_scores)
