import pandas as pd
import numpy as np
from datasets import load_dataset
import openai
from openai import OpenAI
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Union
from PIL import Image
import PIL
import torch

from prompts.claim_decomposition import decomposition_supernova
from prompts.relevance_filtering import relevance_supernova, load_relevance_supernova_prompt
from prompts.expert_alignment import alignment_supernova
from prompts.explanations import vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline, supernova_prompt

from llms import load_model, MyOpenAIModel

from diskcache import Cache
cache = Cache("/shared_data0/chaenyk/llm_cache")

class SupernovaExample:
    def __init__(self,
                 file,
                 time_series_data,
                 ground_truth: Any,
                 llm_label: Any,
                 llm_explanation: str):
        self.file = file
        self.time_series_data = time_series_data
        self.ground_truth = ground_truth
        self.llm_label = llm_label
        self.llm_explanation = llm_explanation
        self.claims = []
        self.relevant_claims = []
        self.alignment_scores = []
        self.alignment_categories = []
        self.alignment_reasonings = []

@cache.memoize()
def query_openai(prompt, model="gpt-4o"):
    with open("../API_KEY.txt", "r") as file:
        api_key = file.read()
    client = OpenAI(api_key=api_key)

    num_tries = 0
    for i in range(3):
        try:
            translation = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                model=model,
            )
            return translation.choices[0].message.content
        except Exception as e:
            num_tries += 1
            print("Try {}; Error: {}".format(str(num_tries), str(e)))     
            time.sleep(3)
    return "ERROR"

def get_llm_output(prompt, images=None, model='gpt-4o'):
    with open("../API_KEY.txt", "r") as file:
        api_key = file.read()
    llm = load_model(model, api_key)
    result = llm([(prompt, images)])[0]
    return result

def format_time_series_for_prompt(time_series_data: Dict[float, Dict[str, Union[float, str]]]) -> str:
    if not time_series_data:
        return "No time-series data provided."
    output_lines = []
    for time in sorted(time_series_data.keys()):
        output_lines.append(f"Time {time}:")
        measurements = time_series_data[time]
        if not measurements:
            output_lines.append("  (No measurements recorded at this time)")
            continue
        for name in sorted(measurements.keys()):
            value = measurements[name]
            value_repr = f"'{value}'" if isinstance(value, str) else str(value)
            output_lines.append(f"  {name}: {value_repr}")

    return "\n".join(output_lines)

def parse_measurement_string(data_string: str) -> Dict[float, Dict[str, Union[float, str]]]:
    measurements_by_time: Dict[float, Dict[str, Union[float, str]]] = {}
    if not isinstance(data_string, str) or not data_string.strip():
        return measurements_by_time 

    parts = [part.strip() for part in data_string.strip().rstrip(';').split(';') if part.strip()]
    for part in parts:
        time_str, measurement_part = part.split(':', 1)
        measurement_name, value_str = measurement_part.split(',', 1)

        time = float(time_str.strip())
        name = measurement_name.strip()
        value_str_cleaned = value_str.strip()
        try:
            value: Union[float, str] = float(value_str_cleaned)
        except ValueError:
            value = value_str_cleaned 

        if time not in measurements_by_time:
            measurements_by_time[time] = {}
        measurements_by_time[time][name] = value
    return measurements_by_time

def get_llm_generated_answer(time_series_data, method: str = "vanilla"):
    if method == "vanilla":
        prompt = supernova_prompt.replace("[BASELINE_PROMPT]", vanilla_baseline)
    elif method == "cot":
        prompt = supernova_prompt.replace("[BASELINE_PROMPT]", cot_baseline)
    elif method == "socratic":
        prompt = supernova_prompt.replace("[BASELINE_PROMPT]", socratic_baseline)
    elif method == "subq" or method == "least_to_most":
        prompt = supernova_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid method: {method}")
    img = time_series_data
    response = get_llm_output(prompt, img, model="gpt-4o")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None

    response_split = [r.strip() for r in response.split("\n") if r.strip() != "" \
        and r.strip().startswith("Explanation:") or r.strip().startswith("Label:")]
    llm_label = response_split[0].split("Label: ")[1].strip()
    explanation = response_split[1].split("Explanation: ")[1].strip()
    return llm_label, explanation

def isolate_individual_features(explanation: str):
    prompt = decomposition_supernova.format(explanation)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")
    return claims

def is_claim_relevant(time_series_text, rating: str, claim: str):
    prompt = relevance_supernova.format(time_series_text, rating, claim)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Relevance:", "").strip()
    response = response.split("\n")
    relevance = response[0].strip()
    reasoning = response[1].replace("Reasoning:", "").strip()
    return relevance, reasoning

def distill_relevant_features(
    example_image: PIL.Image.Image | torch.Tensor | np.ndarray,
    answer: str,
    atomic_claims: list[str],
    model: str = "gpt-4o",
    verbose: bool = False
) -> list[str]:
    """
    Distill the relevant features from the atomic claims.
    """

    prompts = [load_relevance_supernova_prompt(example_image, answer, claim) for claim in atomic_claims]
    llm = load_model(model)
    llm.verbose = True
    results = llm(prompts)

    relevant_claims = [
        claim for claim, result in zip(atomic_claims, results)
        if "relevance: yes" in result.lower()
    ]

    return relevant_claims

def calculate_expert_alignment_score(claim: str):
    prompt = alignment_supernova.format(claim)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Category:", "").strip()
    response = response.split("\n")
    response = [r for r in response if r.strip() != ""]
    category = response[0].strip()
    alignment_score = response[1].replace("Category Alignment Rating:", "").strip()
    try:
        alignment_score = float(alignment_score)
    except:
        print("ERROR: Could not convert alignment score to float")
        print(response)
        alignment_score = 0.0
    reasoning = response[2].replace("Reasoning:", "").strip()
    return category, alignment_score, reasoning
