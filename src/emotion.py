import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from fuzzywuzzy import fuzz
from prompts.claim_decomposition import decomposition_emotion
from prompts.relevance_filtering import relevance_emotion
from prompts.claim_grouping import claim_grouping_emotion
from prompts.expert_category_alignment import category_alignment_emotion
from prompts.explanations import vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline, emotion_prompt

from diskcache import Cache
import functools

from llms import load_model

cache = None  # Will be set inside main

def memoize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if cache is None:
            return func(*args, **kwargs)
        return cache.memoize()(func)(*args, **kwargs)
    return wrapper

default_model = "gpt-5-mini-2025-08-07"


def _load_api_key(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", filename)
    with open(path, "r") as f:
        return f.read().strip()


prompt_dict = {"vanilla": vanilla_baseline,
               "cot": cot_baseline,
               "socratic": socratic_baseline,
               "subq": least_to_most_baseline}
categories_list = [
    "Valence",
    "Arousal",
    "Emotion Words & Emojis",
    "Expressive Punctuation",
    "Humor/Laughter Markers",
    "Confusion Phrases",
    "Curiosity Questions",
    "Surprise Exclamations",
    "Threat/Worry Language",
    "Loss or Let-Down Words",
    "Other-Blame Statements",
    "Self-Blame & Apologies",
    "Aversion Terms",
    "Praise & Compliments",
    "Gratitude Expressions",
    "Affection & Care Words",
    "Self-Credit Statements",
    "Relief Indicators"
]

emotion_labels = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

class EmotionExample:
    def __init__(self, text, ground_truth, llm_label, llm_explanation):
        self.text = text
        self.ground_truth = ground_truth
        self.llm_label = llm_label
        self.llm_explanation = llm_explanation
        self.accuracy = 0.0
        self.claims = []
        self.relevant_claims = []
        self.claims_by_category = []
        self.category_alignment_scores = []
        self.alignment_matrix = []
        self.final_aligned_score = 0.0

    def print(self, verbose=False):
        print("Text: ", self.text)
        print("Ground Truth: ", self.ground_truth)
        print("LLM Label: ", self.llm_label)
        print("LLM Explanation: ", self.llm_explanation)
        print("Claims: ", self.claims)
        print("Relevant Claims: ", self.relevant_claims)
        print("Category Alignment Scores: ", self.category_alignment_scores)
        
        print("Final Alignment Score: ", self.final_aligned_score)
        if(verbose):
            print("Claims By Category: ", self.claims_by_category)
            print("Alignment Matrix: ", self.alignment_matrix)
    
    def to_dict(self):
        return {
            'text': self.text,
            'ground_truth': self.ground_truth,
            'llm_label': self.llm_label,
            'llm_explanation': self.llm_explanation,
            'accuracy': self.accuracy,
            'claims': self.claims,
            'relevant_claims': self.relevant_claims,
            'claims_by_category': self.claims_by_category,
            'category_alignment_scores': self.category_alignment_scores,
            'alignment_matrix': self.alignment_matrix.tolist(),
            'final_aligned_score': self.final_aligned_score
        }
    
@memoize
def query_anthropic(prompt, model="claude-haiku-4-5-20251001"):
    api_key = _load_api_key("Anthropic_API_KEY.txt")
    llm = load_model(model, api_key=api_key)
    out = llm(prompt)
    return out if out else "ERROR"


@memoize
def query_gemini(prompt, model="gemini-2.5-flash"):
    llm = load_model(model)
    out = llm(prompt)
    return out if out else "ERROR"
    

@memoize
def query_openai(prompt, model="gpt-5-mini-2025-08-07"):
    api_key = _load_api_key("API_KEY.txt")
    llm = load_model(model, api_key=api_key)
    out = llm(prompt)
    return out if out else "ERROR"

def get_llm_generated_answer(text: str, baseline: str = "vanilla", model = "gpt-5-mini-2025-08-07"):
    prompt = emotion_prompt.replace("[BASELINE_PROMPT", prompt_dict[baseline]).format(text)

    if("gpt" in model or "o1" in model):
        response = query_openai(prompt, model=model).replace("\n\n", "\n")
    elif("claude" in model):
        response = query_anthropic(prompt, model=model).replace("\n\n", "\n")
    elif("gemini" in model):
        response = query_gemini(prompt, model=model).replace("\n\n", "\n")
    else:
        print("ERROR: Model not supported")
        return None, None

    if response == "ERROR":
        print("Error in querying API for model: ", model)
        return None, None
    response_split = [e for e in response.split("\n") if (e != '' and e.split()[0] in ['Label:', 'Explanation:'])]
    llm_label = response_split[0].split("Label: ")[1].strip().lower()
    explanation = response_split[1].split("Explanation: ")[1].strip()
    try:
        assert(len(explanation) > 10)
        return llm_label, explanation
    except:
        print("ERROR: LLM generated answer is not valid")
        print(response)
        return None, None


def isolate_individual_features(explanation: str, model: str = default_model):
    prompt = decomposition_emotion.format(explanation)
    
    if model == default_model:
        response = query_openai(prompt).replace("\n\n", "\n")
    else:
        llm = load_model(model)
        response = llm([prompt])[0]
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")
    return claims

def is_claim_relevant(text: str, rating: str, claim: str, model: str = default_model):
    prompt = relevance_emotion.format(text, rating, claim)
    if model == default_model:
        response = query_openai(prompt).replace("\n\n", "\n")
    else:
        llm = load_model(model)
        response = llm([prompt])[0]
        
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    try:
        response = response.replace("Relevance:", "").strip()
        response = response.split("\n")
        relevance = response[0].strip()
        reasoning = response[1].replace("Reasoning:", "").strip()
        assert(relevance in ["Yes", "No"])
        assert(len(reasoning) > 10)
    except:
        print("ERROR: Could not determine relevance")
        print(response)
        relevance = "No"
        reasoning = "ERROR"
    return relevance, reasoning


def distill_relevant_features(example: EmotionExample, model: str = default_model):
    relevant_claims = []
    for claim in tqdm(example.claims):
        relevance, reasoning = is_claim_relevant(example.text, example.llm_label, claim, model=model)
        if relevance is None:
            continue
        if relevance == "Yes":
            relevant_claims.append(claim)
    return relevant_claims


def get_claims_by_category(category: str, claims: list[str], model: str = default_model):
    """
    Args:
        category (str): The category to find claims for.
        claims (list[str]): A list of relevant claims.
    Returns:
        list[str]: A list of relevant claims that are related to the category.
    """
    prompt = claim_grouping_emotion.format(category, claims)
    if model == default_model:
        response = query_openai(prompt).replace("\n\n", "\n")
    else:
        llm = load_model(model)
        response = llm([prompt])[0]
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None 
    try:
        assert "RELATED CLAIMS:" in response and "REASONING:" in response
    except:
        print("ERROR: Issue with claim grouping parsing")
        print(response)
        return None
    response = response.split("RELATED CLAIMS:")[1].strip()
    response = response.split("\n")
    response = [r for r in response if r.strip() != "" and r.strip() != "None"]
    return response

def group_claims_by_category(relevant_claims: list[str], model: str = default_model):
    """
    Args:
        relevant_claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        dict[str, list[str]]: A dictionary where the keys are the categories and the values are lists of claims that are aligned with the category.
    """
    claims_by_category = {}
    for category in categories_list:
        related_claims = get_claims_by_category(category, relevant_claims, model=model)
        if related_claims is None:
            claims_by_category[category] = []
            continue
        claims_by_category[category] = related_claims
    return claims_by_category

def calculate_expert_alignment_score_for_category(category: str, claims: list[str], model: str = default_model):
    """
    Args:
        category (str): The category to calculate the alignment score for.
        claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        float: The alignment score for the claims in the category.
    """
    prompt = category_alignment_emotion.format(category, claims)
    if model == default_model:
        response = query_openai(prompt).replace("\n\n", "\n")
    else:
        llm = load_model(model)
        response = llm([prompt])[0]
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.split("Category Alignment Rating:")[1].strip()
    try:
        assert response in ["complete", "partial", "none"]
    except:
        print("ERROR: Issue with alignment score parsing")
        print(response)
        return None
    return response

def calculate_expert_alignment_score(claims_by_category: dict[str, list[str]], model: str = default_model):
    """
    Args:
        claims_by_category (dict[str, list[str]]): A dictionary where the keys are the categories and the values are lists of claims that are aligned with the category.
    Returns:
        dict[str, float]: A dictionary where the keys are the categories and the values are the alignment scores.
    """
    category_alignment_scores = {}
    score_mapping = {"none": 0, "partial": 0.5, "complete": 1}
    for category in claims_by_category.keys():
        if len(claims_by_category[category]) == 0:
            category_alignment_scores[category] = 0
        else:
            category_alignment_score = calculate_expert_alignment_score_for_category(category, claims_by_category[category], model=model)
            if category_alignment_score is None:
                category_alignment_scores[category] = 0
                continue
            category_alignment_scores[category] = score_mapping[category_alignment_score]
    return category_alignment_scores

def make_alignment_matrix(categories, claims, claims_by_category, category_alignment_scores):
    """
    Args:
        categories (list[str]): A list of all expertcategories.
        claims (list[str]): A list of all atomic claims.
        claims_by_category (dict[str, list[str]]): A dictionary where the keys are the categories and the values are lists of claims that are aligned with the category.
        category_alignment_scores (dict[str, float]): A dictionary where the keys are the categories and the values are the alignment scores.
    Returns:
        list[list[float]]: A matrix of alignment scores for the claims in the categories.
    """
    matrix = np.zeros((len(claims), len(categories)))
    for i, claim in enumerate(claims):
        for j, category in enumerate(categories):
            if any(fuzz.ratio(claim, c) > 90 for c in claims_by_category[category]):
                matrix[i, j] = category_alignment_scores[category]
    return matrix


def load_emotion_data():
    emotion_data =  load_dataset("BrachioLab/emotion")
    emotion_data = emotion_data['train'].to_pandas()
    emotion_data['labels'] = emotion_data['labels'].apply(lambda x: [int(i) for i in x])
    emotion_data = emotion_data[emotion_data['text'].apply(lambda x: len(x) > 20)]
    
    #sample 4 examples from each label
    labels = [[x] for x in range(28)]
    emotion_data_sampled = pd.DataFrame()
    for l in labels:
        label_sample = emotion_data[emotion_data['labels'].apply(lambda x: x == l)].sample(4, random_state=11)
        emotion_data_sampled = pd.concat([emotion_data_sampled, label_sample])
    emotion_data = emotion_data_sampled.reset_index(drop=True)
    return emotion_data


def run_pipeline(emotion_data, baseline="vanilla", model="gpt-5.2-pro-2025-12-11"):
    emotion_examples = []
    for idx,row in tqdm(emotion_data.iterrows()):
        label, explanation = get_llm_generated_answer(row['text'], baseline, model)
        if label is None:
            continue
        emotion_examples.append(EmotionExample(
            text=row['text'],
            ground_truth=emotion_labels[int(row['labels'][0])],
            llm_label=label,
            llm_explanation=explanation
        ))
    
    for example in emotion_examples:
        example.accuracy = int(example.ground_truth == example.llm_label)

    print("----- Isolating atomic claims -----")
    for example in tqdm(emotion_examples):
        claims = isolate_individual_features(example.llm_explanation)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]

    print("----- Distilling relevant claims -----")
    for example in tqdm(emotion_examples):
        relevant_claims = distill_relevant_features(example)
        example.relevant_claims = relevant_claims

    print("----- Grouping claims by category -----")
    for example in tqdm(emotion_examples):
        print(example.text)
        claims_by_category = group_claims_by_category(example.relevant_claims)
        example.claims_by_category = claims_by_category

    print("----- Calculating expert alignment scores -----")
    for example in tqdm(emotion_examples):
        category_alignment_scores = calculate_expert_alignment_score(example.claims_by_category)
        example.category_alignment_scores = category_alignment_scores
        example.alignment_matrix = make_alignment_matrix(categories_list, example.claims, example.claims_by_category, example.category_alignment_scores)
        final_aligned_score = example.alignment_matrix.max(axis=-1).mean()
        example.final_aligned_score = final_aligned_score
        
    print("----- Saving results -----")
    data_to_save = [example.to_dict() for example in emotion_examples]
    with open("../results/{}/emotion_{}.json".format(baseline, model), 'w') as f:
        json.dump(data_to_save, f, indent=4)


if __name__ == "__main__":
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/shreyah/FIX-2/gcp-creds.json"

    emotion_data = load_emotion_data().sample(100, random_state=11)
    emotion_data = emotion_data.reset_index(drop=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="vanilla")
    parser.add_argument("--model", type=str, default="gpt-5.2-pro-2025-12-11")
    args = parser.parse_args()
    baseline = args.baseline
    model = args.model
    assert baseline in ["vanilla", "cot", "socratic", "subq"]
    assert model in ["gpt-5.2-pro-2025-12-11", "gpt-5-mini-2025-08-07", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001", "gemini-2.5-pro", "gemini-2.5-flash"]

    #set cache directory
    cache = Cache("/shared_data0/shreyah/llm_cache/emotion/{}".format(baseline))

    run_pipeline(emotion_data, baseline=baseline, model=model)

    