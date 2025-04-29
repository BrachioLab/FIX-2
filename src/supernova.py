import torch
from dataclasses import dataclass
from typing import List
import openai
from openai import OpenAI
import math
import random
import time
from diskcache import Cache
from typing import Tuple
import io
import base64
import os
import re
import json

@dataclass
class Timeseries:
    def __init__(self, time, wv, value):
        self.time = time
        self.wv = wv
        self.value = value

class Image: pass

class Text: pass

cache = Cache("/shared_data0/llm_cachedir")
with open("openai_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)


def get_messages(prompt, system_prompt=None):
    system_message = [
                {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
            ]
    new_message = [
        {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}
    ]
    messages = system_message + new_message
    
    return messages


def text2json(text):
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        json_str = json_str.replace('\\', '\\\\')
        data = json.loads(json_str)
    else:
        data = None
    return data


@cache.memoize()
def get_llm_output(prompt, system_prompt=''):
    """
    prompt: str
    system_prompt: str
    """
    messages = get_messages(prompt, system_prompt)
    response = client.chat.completions.create(
            model='gpt-4o',
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
def get_llm_score(prompt, system_prompt=None) -> Tuple[str, float]:
    if system_prompt is None:
        system_prompt = "Answer only as a YES or NO."
    messages = get_messages(prompt, system_prompt)
    response = client.chat.completions.create(
        model="gpt-4o",
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


def get_llm_generated_answer(
    example: Timeseries
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the timeseries classification task.
    """

    with open('class_example.txt', 'r') as f:
        class_examples = f.read()
        
    time_data = example.time
    wv_data = example.wv
    value_data = example.value
    
    if isinstance(time_data, torch.Tensor):
        time_data = time_data.tolist()
    if isinstance(wv_data, torch.Tensor):
        wv_data = wv_data.tolist()
    if isinstance(value_data, torch.Tensor):
        value_data = value_data.tolist()
    
    data_str = f""" 
    Time data: {time_data} 
    Wavelength data: {wv_data} 
    Value data: {value_data} 
    """
    prompt = f"""You are an expert in analyzing astrophysics data. You will be given supernova data in a time series format. Your task is to predict the class of the supernova by analyzing the data.

The format of the data will be three time series as follows:
Time data: {time_data}
Wavelength data: {wv_data} 
Value data: {value_data} 

Each time series consists of values corresponding to different wavelengths, values, and the time at which they were recorded. 
The wavelength variable corresponds to the type of electromagnetic spectrum used to observe the corresponding value variable which can work as filters. A different wavelength will result in a different range of possible values. By reasoning about the wavelengths and corresponding observed values over multiple timesteps, your task is to classify the type of supernova.
The possible classes are: RR-Lyrae (RRL), peculiar type Ia supernova (SNIa-91bg), type Ia supernova (SNIa), superluminous supernova (SLSN-I), type II supernova (SNII), microlens-single (mu-Lens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar type Ia supernova (SNIax), type Ibc supernova (SNIbc), Mira variable, active galactic nuclei (AGN)

To help you classify the given data, here are examples of data for each class. You can analyze this example data for the given classes to understand what unique features from the data contribute to each classification.

{class_examples}

Your turn! You will now be given data to analyze. To the best of your ability, select which supernova class the data came from. In addition to the class, provide a short paragraph that explains why you chose the selected class. Keep your explanation between 100-200 words and focus on the features of the data you used to make your classification. Your response should be formatted as follows:
Class: <class>
Explanation: <explanation>
Here is the data for you to analyze:
Time data: {time_data}
Wavelength data: {wv_data} 
Value data: {value_data}"""

    system_prompt = "You are an expert in astrophysics."
    
    return get_llm_output(prompt, system_prompt)

    
    messages = [
        {
            "role": "user", 
            "content": f"{prompt}\n\nHere is the time series data for analysis:\n{data_str}"
        }
    ]
    
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages
    )
    return response.choices[0].message.content


def isolate_individual_features(
    explanation: str
):
    """
    Args:
        explanation (str): The LLM-generated reasoning chain of why it gave a specific answer to an example.
        
    Returns:
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    """
    system_prompt_text2claims = """You will be given a paragraph that explains the reasoning behind classifying supernova time series data into one of the following categories: RR-Lyrae (RRL), peculiar type Ia supernova (SNIa-91bg), type Ia supernova (SNIa), superluminous supernova (SLSN-I), type II supernova (SNII), microlens-single (mu-Lens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar type Ia supernova (SNIax), type Ibc supernova (SNIbc), Mira variable, active galactic nuclei (AGN)

Your task is to decompose this explanation into individual claims that are:
Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone. 

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:
INPUT: <example explanation>
OUTPUT:
Claims:
```json
[
    "<claim 1>",
    "<claim 2>",
    ...
]
```

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""
    raw_atomic_claims = get_llm_output(explanation, system_prompt=system_prompt_text2claims)
    return text2json(raw_atomic_claims)


def is_claim_relevant(
    example: Timeseries, answer: str, atomic_claim: str, threshold: float = 0.8
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

    time_data = example.time
    wv_data = example.wv
    value_data = example.value
    
    if isinstance(time_data, torch.Tensor):
        time_data = time_data.tolist()
    if isinstance(wv_data, torch.Tensor):
        wv_data = wv_data.tolist()
    if isinstance(value_data, torch.Tensor):
        value_data = value_data.tolist()
    
    data_str = f""" 
    Time data: {time_data} 
    Wavelength data: {wv_data} 
    Value data: {value_data} 
    """

    system_prompt_is_claim_relevant = f"""You are an expert in astrophysics. Below is a reasoning chain explaining why a specific classification decision was made for a supernova candidate, based on its time-series data.
The dataset includes flux measurements over time across multiple wavelengths for an object labeled {answer}. Use the full context of the dataset, {data_str}, to evaluate whether each claim is relevant. The possible classification categories include: RR-Lyrae (RRL), peculiar type Ia supernova (SNIa-91bg), type Ia supernova (SNIa), superluminous supernova (SLSN-I), type II supernova (SNII), microlens-single (mu-Lens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar type Ia supernova (SNIax), type Ibc supernova (SNIbc), Mira variable, active galactic nuclei (AGN)

    A claim is considered relevant only if both of the following conditions are satisfied:
        (1) It is directly supported by the time-series data (e.g., it refers to trends, changes, or patterns in flux across time and wavelengths).
        (2) It helps explain why the model predicted this specific class (e.g., it highlights characteristics that distinguish this class from others).
        Please only answer YES or NO.
        Here are some examples:
        [Example 1]
        Claim: The dataset represents a time series of observations for a supernova event.
        Answer: NO
        This is a general statement and does not justify any specific classification.
        [Example 2]
        Claim: The time series shows a rapid increase in brightness followed by a gradual decline.
        Answer: YES
        This is a specific flux pattern observable in the data and indicative of certain supernova types such as SNIa.
        [Example 3]
        Claim: Type Ia supernovae are valuable as standard candles for measuring cosmic distances.
        Answer: NO
        This is background knowledge, not derived from the data.
        [Example 4]
        Claim: Specific wavelengths such as 7545.98 Å, 8590.90 Å, and 9710.28 Å are present in the data.
        Answer: NO
        Simply listing wavelengths does not explain a classification.

        Now, determine whether the following claim is relevant to the given the data and classification result."""

    prompt_is_claim_relevant = """Answer:
{}

Atomic Claim:
{}
""".format(answer, atomic_claim)
    completion, prob = get_llm_score(prompt_is_claim_relevant, 
              system_prompt=system_prompt_is_claim_relevant)

    return completion == "yes" and prob >= threshold


def distill_relevant_features(
     example: Timeseries, answer: str, raw_atomic_claims: list[str], threshold: float = 0.8
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
        if is_claim_relevant(example, answer, raw_atomic_claim, threshold):
            atomic_claims.append(raw_atomic_claim)
    return atomic_claims


def calculate_expert_alignment_score(
    atomic_claims: list[str]
):
    
    system_prompt = """You are an expert astrophysicist. Your task is to evaluate how well each of the following claims aligns with known ground truth criteria used in classifying supernovae from flux measurements across time and wavelength.

The ground truth criteria below represent core observational patterns that support the classification of supernovae. These patterns often appear in localized segments of the time series:
1. Nonzero Flux Patterns: Segments where the flux remains consistently above zero indicate successful observations and astrophysical activity. These active regions are critical for detecting transient phenomena, even if the entire time series includes gaps or inactive periods.
2. Multi-wavelength Observability: In segments where multiple wavelength bands are recorded simultaneously, joint spectral and temporal analysis becomes possible. Variation in flux across bands captures spectral diversity, which is often essential for distinguishing between different supernova classes.
3. Temporal Continuity: Smooth or structured flux changes over time (e.g., flat, increasing, or decreasing trend at one or more wavelengths) provides evidence of underlying physical processes like explosions, pulsations, or orbital dynamics. These temporal patterns are key to distinguishing classes.

For each claim, assess how well it semantically and factually aligns with the ground truth indicators above. Avoid focusing on superficial keyword matches and evaluate the actual meaning and evidentiary alignment.

Use the following relevance scale from 1 to 5:
1: Completely contradicts: The claim fundamentally misrepresents or contradicts the criteria used in supernova classification.
2: Mostly contradicts: The claim is largely inconsistent with known indicators or suggests irrelevant patterns.
3: Partially aligns: The claim is related but lacks a clear or accurate connection to ground truth patterns.
4: Mostly aligns: The claim captures a valid classification cue, though with minor vagueness or lack of specificity.
5: Completely aligns: The claim is fully consistent with one or more ground truth indicators and describes meaningful observational patterns useful for classification.
Also provide a brief justification for each score, explaining the reasoning in terms of the observed patterns and their relevance to classification.
Input format:
Claims:
```json
[
    "<claim 1>",
    "<claim 2>",
    ...
]

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
Here are some examples:
[Example 1]
Claim: Observations are recorded at various wavelength overtime.
Score: 2
This claim loosely relates to Ground Truth #2, which emphasizes simultaneous multi-wavelength observations. However, it lacks clarity on whether multiple bands are recorded at the same time or how consistently this occurs, making the alignment weak and incomplete.
[Example 2]
Claim: Consistent and distinct peaks are observed in value data at specific wavelength
Score: 3
While temporal consistency hints at Ground Truth #3, the alignment is only partial due to lack of explicit support for peak-based or wavelength-specific patterns.
[Example 3]
Claim: Variations in intensity over time are typical of the lightcurve evolution of a supernova
Score: 3
This aligns well with ground truth #4 ("temporal continuity"), as it refers to structured temporal variation. But it's a bit generic and leans on domain knowledge which would be true for all supernova time series data.
[Example 4]
Claim: The flux value has a rapid increase and gradual decrease.
Score: 5
This describes a classic lightcurve shape of many supernovae and reflects temporal continuity (Ground Truth #3). It captures meaningful evolution in flux over time, making it highly relevant to classification.
[Example 5]
Claim: Significant fluctuations and peaks in the data can be inferred as part of the light curve of a Type II supernova.
Score: 4
The claim refers to fluctuations and peaks over time, which aligns with the idea of structured evolution in flux, a key temporal pattern that helps distinguish classes like SNII. However, the claim does not specify whether these patterns occur in localized segments of the time series or mention the presence of nonzero flux or uncertainty values.
"""


    prompt = """Claims:
```json
{}
```
""".format(json.dumps(atomic_claims))
          
    alignment_scores = get_llm_output(prompt, system_prompt=system_prompt)
    return text2json(alignment_scores)
