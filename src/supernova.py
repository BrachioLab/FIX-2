import torch
from dataclasses import dataclass
from typing import List
import openai
from openai import OpenAI
import math
import random
import time

@dataclass
class Timeseries:
    def __init__(self, time, wv, value):
        self.time = time
        self.wv = wv
        self.value = value

class Image: pass

class Text: pass


client = OpenAI(api_key="sk-None-ptTj0i2Hx0GJvjOunH0QT3BlbkFJcdhzvU2L1X5HU4F23HWB")

def get_llm_generated_answer(
    example: Timeseries, prompt: str
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
    
    messages = [
        {
            "role": "user", 
            "content": f"{prompt}\n\nHere is the time series data for analysis:\n{data_str}"
        }
    ]
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )
    return response.choices[0].message.content


def isolate_individual_features(
    explanation: str, prompt: str = None
):
    """
    Args:
        explanation (str): The LLM-generated reasoning chain of why it gave a specific answer to an example.
        prompt (str): The prompt to use for the LLM to break down the explanation into atomic claims.
    Returns:
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    """
    if explanation is None:
        raise ValueError("explanation argument cannot be None")
    if prompt is None:
        prompt = f"""Please break the following explanation down into atomic claims. 
        Please list in bullet points:
        - *<point 1>*: xxxx, 
        - *<point 2>*: xxxx,
        ...

        Explanation: {explanation}"""
    
    messages = [
        {
            "role": "user", 
            "content": prompt
        }
    ]
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )
    
    response_text = response.choices[0].message.content
    
    raw_atomic_claims = []
    for line in response_text.split('\n'):
        if line.strip().startswith('-'):
            claim = line.strip()[1:].strip()
            if "*<point" in claim and ">*:" in claim:
                claim = claim.split(">*:", 1)[1].strip()
            raw_atomic_claims.append(claim)
    
    return raw_atomic_claims


def is_claim_relevant(
    example: Timeseries, answer: str, atomic_claims: str, prompt: str = None
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
    if prompt is None:
        prompt = f"""Given the reasoning chain below about how to classify types of supernova, evaluate if each claim is contained in the original timeseries dataset.
        The time series dataset consists of recorded values at different wavelengths over time for Type Ia supernovae (SNIa). Each claim must be checked against this dataset to determine if the necessary information is present.  
        For a claim to be relevant, it must be:
        (1) Directly supported by the data recorded in the time series (i.e., the claim refers to the change and trend in values for each wavelength over time).  
        (2) Answers the question of why the LLM gave the answer for a particular classification decision for this specific example.
        
        For each claim, provide an evaluation in the following format:
        - *<point 1>*: xxxx, judgment: contained / not contained, explain why this claim is (or is not) directly supported by the dataset, referencing specific aspects of the time series data.  
        ..."""
    messages = [
        {"role": "system", "content": "You are an AI that extracts relevant claims based on a given example and answer."},
        {"role": "user", "content": f"""
        Example: {example}
        Answer: {answer}
        Atomic Claims: {atomic_claims}
        
        {prompt}
        """},
    ]
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )
    
    response_text = response.choices[0].message.content
    
    atomic_claims = []
    for line in response_text.split('\n'):
        if line.strip().startswith('-'):
            claim = line.strip()[1:].strip()
            if "*<point" in claim and ">*:" in claim:
                claim = claim.split(">*:", 1)[1].strip()
            atomic_claims.append(claim)
    
    return atomic_claims

def distill_relevant_features(
    example: str | Image | Timeseries, answer: str, atomic_claims: list[str]
):
    """
    Args:
        example (str | Image | timeseries): The input example from a dataset from which to distill the relevant features from.
        answer (str): The LLM-generated answer to the example.
        atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    Returns:
        relevant_claims (list[str]): A list of strings where each string is a relevant claim.
    """
    relevant_claims = []
    
    for claim in atomic_claims:
        # Check if this claim is marked as "contained"
        if "judgment: contained" in claim:
            parts = claim.split(',')
            if len(parts) > 0:
                claim_text = parts[0].strip()
                relevant_claims.append(claim_text)
    
    return relevant_claims


def calculate_expert_alignment_score(
    atomic_claims: list[str], ground_truth: list[str], prompt: str = None
) -> float:
    """
    Computes the individual (and overall) alignment score of all the relevant atomic claims.

    Possibly needs a domain-independent aggregation function.
    Args:
        atomic_claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        1. Alignment score of each individual atomic claims.
        2. Overall alignment score of all the atomic claims.
    """
    if prompt is None:
        prompt = "Check if two scientific claims c1 and c2 are equivalent. Answer only as a YES or NO."
    
    individual_scores = []
    total_score = 0.0
    
    for claim in atomic_claims:
        best_match = None
        best_score = 0.0
        best_result = "NO"
        
        for truth in ground_truth:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"c1: {claim}\nc2: {truth}"}],
                    },
                ],
                response_format={"type": "text"},
                temperature=0,
                max_completion_tokens=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                logit_bias={31958: 100, 14695: 100},  # Biasing towards YES/NO tokens
                logprobs=True,
            )
            
            completion = response.choices[0].logprobs.content[0].token.strip().lower()
            logprob = response.choices[0].logprobs.content[0].logprob
            score = math.exp(logprob)
            
            # Add random delay to prevent rate limiting
            sleep_duration = random.uniform(0.5, 2)
            time.sleep(sleep_duration)
            
            if completion == "yes" and score > best_score:
                best_match = truth
                best_score = score
                best_result = "yes"
        
        # Store the claim, its best match, and the score
        individual_scores.append({
            "claim": claim,
            "matched_ground_truth": best_match,
            "result": best_result,
            "score": best_score
        })
        
        # Add to total score (if there was a match)
        if best_result == "yes":
            total_score += best_score
    
    # Calculate overall score (average of matched scores)
    matched_claims = [s for s in individual_scores if s["result"] == "yes"]
    overall_score = total_score / len(matched_claims) if matched_claims else 0.0
    
    return {
        "individual_scores": individual_scores,
        "overall_score": overall_score
    }