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
    def __init__(self, time, measurement, value):
        self.time = time
        self.measurement = measurement
        self.value = value

class Image: pass

class Text: pass

cache = Cache("/shared_data0/llm_cachedir")
with open("api_key.txt", "r") as f:
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
    example: Timeseries
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the timeseries classification task.
    """
        
    time_data = example.time
    measurement_data = example.measurement
    value_data = example.value
    
    if isinstance(time_data, torch.Tensor):
        time_data = time_data.tolist()
    if isinstance(measurement_data, torch.Tensor):
        measurement_data = measurement_data.tolist()
    if isinstance(value_data, torch.Tensor):
        value_data = value_data.tolist()
    
    data_str = f""" 
    Time data: {time_data} 
    Measurement data: {measurement_data} 
    Value data: {value_data} 
    """
    prompt = f"""You are a medical AI expert specializing in sepsis prediction. You will be provided with time-series Electronic Health Record (EHR) data from the first 30 minutes of an ICU patient's admission. Each entry consists of a timestamp, the name of a measurement or medication, and its corresponding value.

Your task is to determine whether this patient is at high risk of developing sepsis within the next 12 hours.
Clinicians typically assess early warning signs such as:
* The patient’s age (older patients are at higher risk)
* Changes in neurological status (e.g., new confusion or lethargy)
* Disproportionate severity of symptoms compared to initial vital signs (e.g., severe hypotension, hypoxia)
In addition, clinicians define sepsis based on:
* Evidence of infection:  Having a blood culture drawn or receiving antibiotics for at least 4 consecutive days.
* SOFA Score: The SOFA (Sequential Organ Failure Assessment) score assesses dysfunction across six organ systems. A SOFA score of ≥2 within 48 hours of suspected infection suggests sepsis. Look for abnormalities in the following lab values:
 Respiratory: PaO₂/FiO₂ ratio (hypoxia if <300)
 Coagulation: Platelet count (concern if <100,000/µL)
 Liver: Bilirubin levels (elevated if >2 mg/dL)
 Cardiovascular: Hypotension or vasopressor use (dopamine<5µg/kg/min or dobutamine or or epinephrine or norepinephrine)
 Neurological: Glasgow Coma Scale (GCS <12)
 Renal: Creatinine levels (>2.0 mg/dL indicates dysfunction) or low urine output (<500 ml/day)
*Fever: Defined as a body temperature ≥38.0°C (100.4°F)
Your turn! You will now be given data to analyze. To the best of your ability, predict whether or not this person would get sepsis within the next 12 hours. In addition to the prediction, provide a short paragraph that explains why you chose the selected class. Keep your explanation between 100-200 words and focus on the features of the data you used to make your yes or no binary classification. Your response should be formatted as follows:
Prediction: <class>
Explanation: <explanation>

Here is the data for you to analyze:
{data_str}"""

    system_prompt = "You are a medical AI expert specializing in sepsis prediction."
    
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
    system_prompt_text2claims = """You will be given a paragraph that explains the reasoning behind classifying binary classification for sepsis.
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
INPUT: {}"""

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
    measurement_data = example.measurement
    value_data = example.value
    
    if isinstance(time_data, torch.Tensor):
        time_data = time_data.tolist()
    if isinstance(measurement_data, torch.Tensor):
        measurement_data = measurement_data.tolist()
    if isinstance(value_data, torch.Tensor):
        value_data = value_data.tolist()

    time_data = example.time
    measurement_data = example.measurement
    value_data = example.value
    
    data_str = f""" 
    Time data: {time_data} 
    Measurement data: {measurement_data} 
    Value data: {value_data} 
    """

    system_prompt_is_claim_relevant = f""" You are a medical AI expert specializing in sepsis prediction.
 You will be given:
A claim about why a patient was predicted to either develop sepsis within 12 hours or not.
A label (yes or no) indicating whether the patient was predicted to develop sepsis.
A time-series EHR data from the first 30 minutes of the ICU admission that includes the name of a measurement or medication and its corresponding value over time.


 Use the full context of the data, {data_str} and the answer {answer} for this patient to get sepsis within 12 hours to evaluate whether each claim is relevant.

    A claim is considered relevant only if both of the following conditions are satisfied:
        (1) It is directly supported by the time-series data (e.g.,reference to a specific value, trend, or change in a measurement such as heart rate, temperature over time).
        (2) It helps explain why the model predicted this specific class (e.g., it contributes to explaining why the model predicted the specified class (yes/no), based on known sepsis indicators (e.g., organ dysfunction, suspected infection, SOFA criteria, vital sign abnormalities)).

        Please only answer YES or NO.
        Here are some examples:
        [Example 1]
        Claim: The dataset represents a time series of a person in ICU.
        Answer: NO
        This is a general statement and does not justify any specific classification.
        [Example 2]
        Claim: The patient is 50 years old, which is a moderate risk factor for sepsis.
        Answer: YES
        The age is directly supported by the data and is a known clinical risk factor for sepsis.
        [Example 3]
        Claim: The patient's condition warrants close monitoring and further investigation for infection.
        Answer: YES
        This claim includes signs of possible infection, and infection aligns with the model's rationale for predicting sepsis.
        [Example 4]
        Claim: The patient exhibits several risk factors and early warning signs for sepsis.
        Answer: NO
        This is too vague. It does not specify what the risk factors or warning signs are, nor does it directly reference any values or patterns from the data. 

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
    
    system_prompt = """ You are a medical AI expert specializing in sepsis prediction. Your task is to evaluate how well each of the following claims aligns with the established clinical ground truth criteria for predicting whether a patient will develop sepsis within 12 hours. The input data consists of time-series EHR data from the first 30 minutes of ICU admission, including the name of each measurement or medication and its corresponding value over time.
The clinical ground truth criteria you should use for evaluation are:
*Patient’s Age: Older patients are at higher risk.
*Neurological Changes: New confusion, decreased Glasgow Coma Scale (GCS), or lethargy.
*Severity Relative to Vital Signs: Disproportionately severe symptoms, such as severe hypotension or hypoxia.
*Evidence of Infection: Blood culture drawn or receipt of antibiotics for at least 4 consecutive days.
*SOFA Score Indicators (Sequential Organ Failure Assessment), particularly:
 Respiratory: PaO₂/FiO₂ ratio (<400 concerning; <300, <200 more severe).
 Coagulation: Platelet count (<150,000/µL concerning; <100,000 or <50,000 worse).
 Liver: Bilirubin (>1.2 mg/dL elevated; >2 or >6 increasingly worse).
 Cardiovascular: Hypotension or vasopressor use (e.g., dopamine, norepinephrine).
 Neurological: Glasgow Coma Scale (GCS) <15.
 Renal: Elevated creatinine (>1.2 mg/dL) or low urine output.
*Fever: Body temperature ≥38.0°C (100.4°F).

For each claim, assess how well its meaning and evidence align with the clinical ground truth indicators above. Focus on semantic and clinical correctness, not just superficial keyword matches. Your evaluation should consider whether the claim genuinely provides relevant information for predicting sepsis based on the criteria.

Use the following relevance scale from 1 to 5:
1: Completely contradicts: The claim misrepresents or contradicts known sepsis indicators.
2: Mostly contradicts: The claim is largely inconsistent with known indicators or suggests irrelevant patterns.
3: Partially aligns: The claim has some relevance but lacks a clear or accurate connection.
4: Mostly aligns: The claim reflects a valid indicator but may lack precision.
5: Completely aligns: The claim is fully consistent with one or more ground truth indicators and describes meaningful observational patterns useful for prediction.
Also briefly justify each score by explaining your reasoning in terms of the observed clinical patterns and their relevance to sepsis prediction.
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
Claim: The patient is a male aged 20–40 years old.
Score: 5
Justification: Younger patients are at lower risk of sepsis. Aligns with ground truth if the model predicts low risk.
[Example 2]
Claim: The patient's temperature is 99.6°F, close to the fever threshold.
Score: 5
Fever is a ground truth indicator, and a borderline temperature may suggest an upward trend which the patient might have sepsis within 12 hours.
[Example 3]
Claim: The patient's pulse oximetry is 92%, indicating possible hypoxia.
Score: 5
Hypoxia is directly related to SOFA’s respiratory component.
[Example 4]
Claim: An elevated white blood cell count suggests an inflammatory or infectious process.
Score: 3
WBC elevation supports infection suspicion but is not directly part of SOFA or infection criteria.
[Example 5]
Claim: The patient's glucose level is significantly elevated at 336 mg/dL.
Score: 2
While possibly relevant, glucose is not part of the ground truth criteria.
[Example 6]
Claim: The patient's condition warrants close monitoring and further investigation for infection.
Score: 1
Too vague and lacks specific reference to any ground truth indicator.
"""

    prompt = """Claims:
```json
{}
```
""".format(json.dumps(atomic_claims))
          
    alignment_scores = get_llm_output(prompt, system_prompt=system_prompt)
    return text2json(alignment_scores)