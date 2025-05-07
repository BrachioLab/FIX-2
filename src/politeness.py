import pandas as pd
import numpy as np
from datasets import load_dataset
import openai
from openai import OpenAI
import time
from tqdm import tqdm

from prompts.claim_decomposition import decomposition_politeness
from prompts.relevance_filtering import relevance_politeness
from prompts.expert_alignment import alignment_politeness
from prompts.explanations import vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline, politeness_prompt

from diskcache import Cache
cache = Cache("/shared_data0/shreyah/llm_cache")

prompt_dict = {"vanilla": vanilla_baseline,
               "cot": cot_baseline,
               "socratic": socratic_baseline,
               "subq": least_to_most_baseline}

class PolitenessExample:
    def __init__(self, utterance, ground_truth, llm_score, llm_explanation):
        self.utterance = utterance
        self.ground_truth = ground_truth
        self.llm_score = llm_score
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

def get_llm_generated_answer(utterance: str, baseline: str = "vanilla"):
    prompt = politeness_prompt.replace("[BASELINE_PROMPT", prompt_dict[baseline]).format(utterance)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    rating = response.split("\n")[0].split("Rating: ")[1].strip()
    explanation = response.split("\n")[1].split("Explanation: ")[1].strip()
    return rating, explanation

def isolate_individual_features(explanation: str):
    prompt = decomposition_politeness.format(explanation)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")
    return claims

def is_claim_relevant(utterance: str, rating: str, claim: str):
    prompt = relevance_politeness.format(utterance, rating, claim)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Relevance:", "").strip()
    response = response.split("\n")
    relevance = response[0].strip()
    reasoning = response[1].replace("Reasoning:", "").strip()
    return relevance, reasoning


def distill_relevant_features(example: PolitenessExample):
    relevant_claims = []
    for claim in tqdm(example.claims):
        relevance, reasoning = is_claim_relevant(example.utterance, example.llm_score, claim)
        if relevance is None:
            continue
        if relevance == "Yes":
            relevant_claims.append(claim)
    return relevant_claims

def calculate_expert_alignment_score(claim: str):
    prompt = alignment_politeness.format(claim)
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