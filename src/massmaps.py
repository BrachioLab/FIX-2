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
    
    user_message = {'role': 'user', 'content': image_payload + [
            {'type': 'text', 'text': user_prompt.format(user_text)}
        ]
        }
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

@cache.memoize()
def get_llm_output_from_messages(messages, model='gpt-4o'):
    """
    messages: list of messages
    """
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
    system_prompt_text2claims = """You will be given a paragraph that explains the reasoning behind using weak lensing map to predict two cosmological parameters Omega_m (which captures the average energy density of all matter in the universe (relative to the total energy density which includes radiation and dark energy)) and sigma_8 (which describes the fluctuation of matter distribution).

Your task is to decompose this explanation into individual claims that are:
Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone. 

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:
INPUT: The weak lensing map shows a distribution of matter density with varying colors indicating different density levels. The presence of several yellow pixels suggests the existence of clusters, indicating regions of high matter density. These clusters are crucial for estimating Omega_m, as they reflect the total matter content in the universe. The blue areas represent voids, indicating low-density regions. The balance between these voids and clusters helps in estimating sigma_8, which measures the amplitude of matter fluctuations. The map shows a moderate number of clusters and voids, suggesting a balanced distribution of matter. This balance implies a moderate value for Omega_m, as there is neither an overwhelming presence of clusters nor voids. The presence of distinct clusters and voids also suggests a moderate value for sigma_8, indicating a typical level of matter fluctuation amplitude.
OUTPUT: 
```json
[
"The weak lensing map shows a distribution of matter density with varying colors indicating different density levels.",
"The presence of several yellow pixels suggests the existence of clusters, indicating regions of high matter density.",
"The present clusters are crucial for estimating Omega_m, as they reflect the total matter content in the universe.",
"The blue areas on the map represent voids, indicating low-density regions.",
"The balance between voids and clusters on the map helps in estimating sigma_8, which measures the amplitude of matter fluctuations.",
"The map shows a moderate number of clusters and voids, suggesting a balanced distribution of matter.",
"A balanced distribution of matter implies a moderate value for Omega_m, as there is neither an overwhelming presence of clusters nor voids.",
"The presence of distinct clusters and voids suggests a moderate value for sigma_8, indicating a typical level of matter fluctuation amplitude."
]
```

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
""".format(explanation)


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
    
    system_prompt_is_claim_relevant = f"""You are an expert in cosmology. Below is a reasoning chain explaining why a specific prediction decision was made for a supernova candidate, based on its time-series data.
The data is a weak lensing map, as shown in the image, which is the spatial distribution of matter density in the universe, for Omega_m={answer['Omega_m']}, sigma_8={answer['sigma_8']}. Use the full context of the data in the image, to evaluate whether each claim is relevant. The possible values for Omega_m is between [0.1, 0.5], and for sigma_8 is between [0.4, 1.4].

Here is the colormap used to create the visualization of this weak lensing map for your reference for mapping the image to numbers:
custom_cmap = get_custom_colormap([
            (-3, "blue"),   # Blue at -3 std
            (0, "gray"),   # Gray at 0 (below this is void)
            (2.9, "red"),   # Red at 2.9 std (this is the upperbound for not being a cluster)
            (3, "yellow"),   # Yellow at 3 std (above this is cluster)
            (20, "white")  # White at 20 std
        ])

A claim is considered relevant only if both of the following conditions are satisfied:
    (1) It is directly supported by the image data (e.g., it refers to trends, changes, or patterns in intensity across different spatial positions in the weak lensing map).
    (2) It helps explain why the model predicted these specific values for Omega_m and sigma_8 (e.g., it highlights characteristics that distinguish these values from other potential values).
    Please only answer YES or NO.
    Here are some examples:
    [Example 1]
    Claim: The dataset represents the spatial distribution of matter density in the universe.
    Answer: NO
    This is a general statement and does not justify any specific prediction.
    [Example 2]
    Claim: The weak lensing map shows several yellow pixels close to each other on the left side, suggesting the existence of high-density regions or clusters.
    Answer: YES
    This is a specific cosmological structure observable in the data and indicative of cosmological parameters such as sigma_8.
    [Example 3]
    Claim: Voids are large low density regions in space.
    Answer: NO
    This is background knowledge, not derived from the data.
    [Example 4]
    Claim: There is a gray pixel in the upper left corner with value 6.2992e-04 in the data.
    Answer: NO
    Simply listing pixel values does not explain a prediction.

    Now, determine whether the following claim is relevant to the given the data in the provided image and prediction result.
"""

    prompt_is_claim_relevant = """Claim:
{}
"""

    image_pil = massmap_to_pil_norm(example)

    system_message = get_system_message(system_prompt_is_claim_relevant)
    # def get_few_shot_user_assistant_messages(images_list, user_text_list, assistant_text_list, user_prompt, assistant_prompt):
    
    # def get_example_message(image, user_text, user_prompt, assistant_text=None, assistant_prompt=None):

# def get_example_message(image, user_text, user_prompt, assistant_text=None, assistant_prompt=None):
    user_message, assistant_message = get_example_message(image_pil, user_text=atomic_claim, user_prompt=prompt_is_claim_relevant)

    messages = system_message + [user_message]

    completion = get_llm_output_from_messages(messages)

    try:
        return completion.split('\n')[0].strip().replace('Answer for relevance: ', '').lower() == "yes"
    except:
        return None

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
