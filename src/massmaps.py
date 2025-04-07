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

cache = Cache("/shared_data0/llm_cachedir")

openai.api_key = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI(api_key=openai.api_key)

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
def get_llm_output(prompt, images=None, system_prompt=''):
    """
    prompt: str
    images: list of PIL images
    system_prompt: str
    """
    messages = get_messages(prompt, images, system_prompt)
    response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            response_format={'type': 'text'},
            temperature=0,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    return response.choices[0].message.content


@cache.memoize()
def get_llm_score(prompt, images=None, system_prompt=None) -> Tuple[str, float]:
    if system_prompt is None:
        system_prompt = "Answer only as a YES or NO."
    messages = get_messages(prompt, images, system_prompt)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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


def get_llm_generated_answer(
    example: str | torch.Tensor, #Image | Timeseries,
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the emotion classification task.
    """
    prompt = """This is an image of a weak lensing map. Please predict Omega_m and sigma_8 values from this and provide a reasoning chain for what interpretable cosmological features you see from this map that you use to make such predictions. Provide a short paragraph that is around 100-200 words."""
    system_prompt = "You are an expert cosmologist."
    
    image_pil = massmap_to_pil(example)
    return get_llm_output(prompt, [image_pil], system_prompt)


def isolate_individual_features(
    explanation: str
):
    """
    Args:
        explanation (str): The LLM-generated reasoning chain of why it gave a specific answer to an example.
        
    Returns:
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    """
    system_prompt_text2claims = """You are an expert cosmologist. This is the explanation and answer for predicting from mass maps. Please break it down into atomic claims.
Output format:
Claims:
```json
[
    "<claim 1>",
    "<claim 2>",
    ...
]
```
"""

    raw_atomic_claims = get_llm_output(explanation, system_prompt=system_prompt_text2claims)
    # return raw_atomic_claims
    return text2json(raw_atomic_claims)


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
    
    system_prompt_is_claim_relevant = """You are an expert cosmologist. You need to check if a claim is relevant to the image of weak lensing map or not.
For a claim to be relevant, it must be:
(1) Supported by the example.
(2) Answers the question of why the LLM gave the answer it did for this specific example.

Please only answer YES or NO.
"""

    prompt_is_claim_relevant = """Answer:
{}

Atomic Claim:
{}
""".format(answer, atomic_claim)

    image_pil = massmap_to_pil(example)
    completion, prob = get_llm_score(prompt_is_claim_relevant, 
              images=[image_pil], 
              system_prompt=system_prompt_is_claim_relevant)

    return completion == "yes" and prob >= threshold
    # raise NotImplementedError()


def distill_relevant_features(
    example: str | torch.Tensor,
    answer: str,
    atomic_claims: list[str],
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
        if is_claim_relevant(example, llm_generated_answer, raw_atomic_claim):
            atomic_claims.append(raw_atomic_claim)
    return atomic_claims

def calculate_expert_alignment_score(
    atomic_claims: list[str],
) -> AlignmentScores:
    """
    Computes the individual (and overall) alignment score of all the relevant atomic claims.

    Possibly needs a domain-independent aggregation function.
    Args:
        atomic_claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        1. Alignment score of each individual atomic claims.
        2. Overall alignment score of all the atomic claims.
    """
    
    system_prompt = """You are an expert cosmologist. You need to check if each claim is aligned with domain knowledge of cosmology. Computes the individual (and overall) alignment score of all the atomic claims.

Input format:
Claims:
```json
[
    "<claim 1>",
    "<claim 2>",
    ...
]
```

Output format:
Scores:
```json
{
    "alignment_scores": {
        {"claim": "<claim 1>", "score": <alignment score ranging from 1 to 5>},
        {"claim": "<claim 2>", "score": <alignment score ranging from 1 to 5>},
        ...
    },
    "total_score": <total alignment score ranging from 1 to 5>
}
```
"""

    prompt = """Claims:
```json
{}
```
""".format(json.dumps(atomic_claims))
          
    alignment_scores = get_llm_output(prompt, system_prompt=system_prompt)
    return text2json(alignment_scores)


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
