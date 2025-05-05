import pandas as pd
import numpy as np
from datasets import load_dataset
import openai
from openai import OpenAI
import time
from tqdm import tqdm

from prompts.claim_decomposition import decomposition_emotion
from prompts.relevance_filtering import relevance_emotion
from prompts.expert_alignment import alignment_emotion
from prompts.explanations import vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline, emotion_prompt

from diskcache import Cache
cache = Cache("/shared_data0/shreyah/llm_cache")

class EmotionExample:
    def __init__(self, text, ground_truth, llm_label, llm_explanation):
        self.text = text
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

def get_llm_generated_answer(text: str):
    prompt = emotion_prompt.replace("[BASELINE_PROMPT]", vanilla_baseline).format(text)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response_split = [e for e in response.split("\n") if (e != '' and e.split()[0] in ['Label:', 'Explanation:'])]
    llm_label = response_split[0].split("Label: ")[1].strip()
    explanation = response_split[1].split("Explanation: ")[1].strip()
    return llm_label, explanation


def isolate_individual_features(explanation: str):
    prompt = decomposition_emotion.format(explanation)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")
    return claims

def is_claim_relevant(text: str, rating: str, claim: str):
    prompt = relevance_emotion.format(text, rating, claim)
    response = query_openai(prompt)
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Relevance:", "").strip()
    response = response.split("\n")
    relevance = response[0].strip()
    reasoning = response[1].replace("Reasoning:", "").strip()
    return relevance, reasoning


def distill_relevant_features(example: EmotionExample):
    relevant_claims = []
    for claim in tqdm(example.claims):
        relevance, reasoning = is_claim_relevant(example.text, example.llm_label, claim)
        if relevance is None:
            continue
        if relevance == "Yes":
            relevant_claims.append(claim)
    return relevant_claims

def calculate_expert_alignment_score(claim: str):
    prompt = alignment_emotion.format(claim)
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