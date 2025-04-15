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
client = OpenAI(api_key="")


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
        # Escape single backslashes
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
    example: Timeseries, answer: str
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the timeseries classification task.
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
    prompt = f"""Analyze this supernova time series data:\n{data_str}
    This dataset represents astrophysical observations, where each time series consists of values corresponding to different wavelengths and the time at which these values were recorded. Different wavelength corresponds to different electromagnetic spectrum at which observations of the value of supernova's light were taken. Classify the type of supernova(e.g., SNIa, SNIbc, SNIax, SNII, RRL, PISN) based on the information from this time series dataset. Provide a reasoning chain for what interpretable time series astrophysics features you see from this data that you use to make such predictions. Provide a short paragraph that is around 100-200 words."""
    system_prompt = "You are an expert astrophysics."
    
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
    system_prompt_text2claims = """You are an expert astrophysics. This is the explanation and answer for classifying the supernova time series data among various classes such as SNIa, SNIbc, SNIax, SNII, RRL, PISN. Please break it down into atomic claims.
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

    system_prompt_is_claim_relevant = f"""You are an expert astrophysics. Given the reasoning chain below about how to classify types of supernova, evaluate if each claim is contained in the original timeseries dataset {data_str}.
        The time series dataset consists of recorded values at different wavelengths over time for {answer}. Each claim must be checked against this dataset to determine if the necessary information is present.  
        For a claim to be relevant, it must be:
        (1) Directly supported by the data recorded in the time series (i.e., the claim refers to the change and trend in values for each wavelength over time).  
        (2) Answers the question of why the LLM gave the answer for a particular classification decision for this specific example. (i.e., it directly relates to the trend that is relevant to classifying supernovae classes).
        
        Please only answer YES or NO."""
    
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
    atomic_claims: list[str], ground_truth: list[str]
):
    """
    Computes the individual (and overall) alignment score of all the relevant atomic claims.

    Possibly needs a domain-independent aggregation function.
    Args:
        atomic_claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        1. Alignment score of each individual atomic claims.
        2. Overall alignment score of all the atomic claims.
    """
    
    system_prompt = """You are an expert astrophysicist. You need to check if each claim is aligned with the provided ground truth statements about astrophysics, especially regarding supernovae.
    
For each claim, determine its alignment with the ground truth on a scale from 1 to 5:
1: Completely contradicts ground truth
2: Mostly contradicts ground truth
3: Partially aligns with ground truth
4: Mostly aligns with ground truth
5: Completely aligns with ground truth

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
