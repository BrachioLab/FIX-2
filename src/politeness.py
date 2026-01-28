import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import json
from tqdm import tqdm
from fuzzywuzzy import fuzz
import re
import argparse
from llms import load_model
from prompts.claim_decomposition import decomposition_politeness
from prompts.relevance_filtering import relevance_politeness
from prompts.expert_alignment import alignment_politeness
from prompts.claim_grouping import claim_grouping_politeness
from prompts.expert_category_alignment import category_alignment_politeness
from prompts.explanations import vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline, politeness_prompt

from diskcache import Cache
import functools
cache = None  # Will be set inside main


def memoize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if cache is None:
            return func(*args, **kwargs)
        return cache.memoize()(func)(*args, **kwargs)
    return wrapper

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/shreyah/FIX-2/gcp-creds.json"

def _load_api_key(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", filename)
    with open(path, "r") as f:
        return f.read().strip()


prompt_dict = {"vanilla": vanilla_baseline,
               "cot": cot_baseline,
               "socratic": socratic_baseline,
               "subq": least_to_most_baseline}

categories_list = [
    "Honorifics and Formal Address",
    "Courteous Politeness Markers",
    "Gratitude Expressions",
    "Apologies and Acknowledgment of Fault",
    "Indirect and Modal Requests",
    "Hedging and Tentative Language",
    "Inclusive Pronouns and Group-Oriented Phrasing",
    "Greeting and Interaction Initiation",
    "Compliments and Praise",
    "Softened Disagreement or Face-Saving Critique",
    "Urgency or Immediacy of Language",
    "Avoidance of Profanity or Negative Emotion",
    "Bluntness and Direct Commands",
    "Empathy or Emotional Support",
    "First-Person Subjectivity Markers",
    "Second Person Responsibility or Engagement",
    "Questions as Indirect Strategies",
    "Discourse Management with Markers",
    "Ingroup Language and Informality"
]

class PolitenessExample:
    def __init__(self, utterance, ground_truth, llm_score, llm_explanation):
        self.utterance = utterance
        self.ground_truth = ground_truth
        self.llm_score = llm_score
        self.llm_explanation = llm_explanation
        self.mse = 0.0
        self.claims = []
        self.relevant_claims = []
        self.claims_by_category = []
        self.category_alignment_scores = []
        self.alignment_matrix = []
        self.final_aligned_score = 0.0
       
    
    def print(self, verbose=False):
        print("Utterance: ", self.utterance)
        print("Ground Truth: ", self.ground_truth)
        print("LLM Score: ", self.llm_score)
        print("LLM Explanation: ", self.llm_explanation)
        print("Claims: ", self.claims)
        print("Relevant Claims: ", self.relevant_claims)
        print("Alignment Scores: ", self.alignment_scores)
        
        print("Final Alignment Score: ", self.final_alignment_score)
        if(verbose):
            print("Alignment Categories: ", self.alignment_categories)
            print("Alignment Reasonings: ", self.alignment_reasonings)
    
    def to_dict(self):
        return {
            'utterance': self.utterance,
            'ground_truth': self.ground_truth,
            'llm_score': self.llm_score,
            'llm_explanation': self.llm_explanation,
            'mse': self.mse,
            'claims': self.claims,
            'relevant_claims': self.relevant_claims,
            'claims_by_category': self.claims_by_category,
            'category_alignment_scores': self.category_alignment_scores,
            'alignment_matrix': self.alignment_matrix.tolist(),
            'final_aligned_score': self.final_aligned_score,
        }


@memoize
def query_anthropic(prompt, model="claude-haiku-4-5-20251001"):
    api_key = _load_api_key("Anthropic_API_KEY.txt")
    llm = load_model(model, api_key=api_key)
    out = llm(prompt)
    return out if out else "ERROR"

@memoize
def query_gemini(prompt, model="gemini-2.5-flash"):
    #make sur
    llm = load_model(model)
    out = llm(prompt)
    return out if out else "ERROR"

@memoize
def query_openai(prompt, model="gpt-5-mini-2025-08-07"):
    api_key = _load_api_key("API_KEY.txt")
    llm = load_model(model, api_key=api_key)
    out = llm(prompt)
    return out if out else "ERROR"


def get_llm_generated_answer(utterance: str, baseline: str = "vanilla", model: str = "gpt-5-mini-2025-08-07"):
    """
    Constructs a baseline-specific politeness prompt for an utterance and queries the LLM.

    Parses the response to extract a politeness rating and explanation.

    Args:
        utterance (str): The input text to evaluate.
        baseline (str): The prompting strategy to use ("vanilla", "cot", "socratic", "subq").

    Returns:
        Tuple[float, str] or None: The LLM's rating and explanation, or None on error.
    """
    prompt = politeness_prompt.replace("[BASELINE_PROMPT", prompt_dict[baseline]).format(utterance)
    
    if("gpt" in model or "o1" in model):
        response = query_openai(prompt, model=model).replace("\n\n", "\n")
    elif("claude" in model):
        response = query_anthropic(prompt, model=model).replace("\n\n", "\n")
    elif("gemini" in model):
        response = query_gemini(prompt, model=model).replace("\n\n", "\n")
    else:
        print("ERROR: Model not supported")
        return None

    if response == "ERROR":
        print("Error in querying API for model: ", model)
        return None
    
    match = re.search(r"Rating: (.*)\nExplanation: (.*)", response)
    if not match:
        print("ERROR: LLM generated answer is not valid")
        print(response)
        return None, None
    response = match.group(0)
    rating = response.split("\n")[0].split("Rating: ")[1].split(":")[0].strip()
    explanation = response.split("\n")[1].split("Explanation: ")[1].strip()  
    try:      
        rating = float(rating)
        assert(len(explanation) > 10)
        return rating, explanation
    except:
        print("ERROR: LLM generated answer is not valid")
        print(response)
        return None, None
          

def isolate_individual_features(explanation: str):
    """
    Uses a decomposition prompt to extract individual claims/features from a model explanation.

    Args:
        explanation (str): The full explanation text to decompose.

    Returns:
        List[str] or None: A list of extracted claims, or None on error.
    """
    prompt = decomposition_politeness.format(explanation)
    response = query_openai(prompt).replace("\n\n", "\n")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("OUTPUT:", "").strip()
    claims = response.split("\n")
    return claims


def is_claim_relevant(utterance: str, rating: str, claim: str):
    """
    Determines whether a given claim is relevant to a specific utterance and rating.

    Args:
        utterance (str): The input utterance.
        rating (str): The LLM's politeness rating.
        claim (str): A single claim extracted from the LLM explanation.

    Returns:
        Tuple[str, str] or None: Relevance ("Yes"/"No") and the reasoning behind it, or None on error.
    """
    prompt = relevance_politeness.format(utterance, rating, claim)
    response = query_openai(prompt).replace("\n\n", "\n")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Relevance:", "").strip()
    response = response.split("\n")
    try:
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


def distill_relevant_features(example: PolitenessExample):
    """
    Filters the claims of a PolitenessExample to retain only those deemed relevant by the LLM.

    Iterates through the claims and applies the relevance prompt to each.

    Args:
        example (PolitenessExample): The example containing claims to evaluate.

    Returns:
        List[str]: A list of relevant claims.
    """
    relevant_claims = []
    for claim in tqdm(example.claims):
        relevance, reasoning = is_claim_relevant(example.utterance, example.llm_score, claim)
        if relevance is None:
            continue
        if relevance == "Yes":
            relevant_claims.append(claim)
    return relevant_claims

def get_claims_by_category(category: str, claims: list[str]):
    """
    Args:
        category (str): The category to find claims for.
        claims (list[str]): A list of relevant claims.
    Returns:
        list[str]: A list of relevant claims that are related to the category.
    """
    prompt = claim_grouping_politeness.format(category, claims)
    response = query_openai(prompt).replace("\n\n", "\n")
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

def group_claims_by_category(relevant_claims: list[str]):
    """
    Args:
        relevant_claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        dict[str, list[str]]: A dictionary where the keys are the categories and the values are lists of claims that are aligned with the category.
    """
    claims_by_category = {}
    for category in categories_list:
        related_claims = get_claims_by_category(category, relevant_claims)
        if related_claims is None:
            claims_by_category[category] = []
            continue
        claims_by_category[category] = related_claims
    return claims_by_category

def calculate_expert_alignment_score_for_category(category: str, claims: list[str]):
    """
    Args:
        category (str): The category to calculate the alignment score for.
        claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        float: The alignment score for the claims in the category.
    """
    prompt = category_alignment_politeness.format(category, claims)
    response = query_openai(prompt).replace("\n\n", "\n")
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

def calculate_expert_alignment_score(claims_by_category: dict[str, list[str]]):
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
            category_alignment_score = calculate_expert_alignment_score_for_category(category, claims_by_category[category])
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


def load_politeness_data():
    """
    Loads and samples multilingual politeness data from the HuggingFace dataset.

    For each combination of language and rounded politeness class, samples 6 examples
    to ensure balanced coverage across language and politeness score bins.

    Returns:
        pandas.DataFrame: A sampled subset of the multilingual politeness dataset.
    """
    languages = ['english', 'spanish', 'chinese', 'japanese']
    classes = [-2, -1, 0, 1, 2]

    politeness_data =  load_dataset("BrachioLab/multilingual_politeness")
    politeness_data = politeness_data['train'].to_pandas()

    sampled_data = pd.DataFrame()
    for lang in languages:
        for cls in classes:
            politeness_data =  load_dataset("BrachioLab/multilingual_politeness")
            politeness_data = politeness_data['train'].to_pandas()
            politeness_data = politeness_data[politeness_data['language'] == lang]
            politeness_data = politeness_data[np.round(politeness_data['politeness']) == cls]
            politeness_data = politeness_data.sample(5, random_state=11).reset_index(drop=True)
            sampled_data = pd.concat([sampled_data, politeness_data], ignore_index=True)
    sampled_data = sampled_data.reset_index(drop=True)
    return sampled_data

def run_pipeline(politeness_data, baseline="vanilla", model="gpt-5.2-pro-2025-12-11"):
    """
    Executes the full politeness evaluation pipeline on a dataset of utterances.

    The pipeline consists of:
    1. Generating LLM scores and explanations.
    2. Extracting claims from explanations.
    3. Filtering relevant claims.
    4. Grouping the relevant claims by category.
    5. Calculating the alignment score for each category.
    6. Making a matrix of alignment scores for the claims in the categories.
    7. Aggregating the alignment scores from the matrix.
    8. Saving the results to a JSON file.

    Args:
        politeness_data (pandas.DataFrame): The input dataset with columns including 'Utterance' and 'politeness'.
        baseline (str): The prompting baseline strategy to use (e.g., "vanilla", "cot").
    
    Returns:
        None: Results are saved as a JSON file under `../results/{baseline}/politeness.json`.
    """
    politeness_examples = []
    for idx,row in tqdm(politeness_data.iterrows()):
        rating, explanation = get_llm_generated_answer(row['Utterance'], baseline, model)
        if rating is None:
            continue
        politeness_examples.append(PolitenessExample(
            utterance=row['Utterance'],
            ground_truth=float(row['politeness']) + 3,
            llm_score=rating,
            llm_explanation=explanation
        ))
    
    for example in politeness_examples:
        example.mse = (example.ground_truth - example.llm_score) ** 2

    print("----- Isolating atomic claims -----")
    for example in tqdm(politeness_examples):
        claims = isolate_individual_features(example.llm_explanation)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]

    print("----- Distilling relevant claims -----")
    for example in tqdm(politeness_examples):
        relevant_claims = distill_relevant_features(example)
        example.relevant_claims = relevant_claims

    print("----- Grouping claims by category -----")
    for example in tqdm(politeness_examples):
        print(example.utterance)
        claims_by_category = group_claims_by_category(example.relevant_claims)
        example.claims_by_category = claims_by_category

    print("----- Calculating expert alignment scores -----")
    for example in tqdm(politeness_examples):
        category_alignment_scores = calculate_expert_alignment_score(example.claims_by_category)
        example.category_alignment_scores = category_alignment_scores
        example.alignment_matrix = make_alignment_matrix(categories_list, example.claims, example.claims_by_category, example.category_alignment_scores)
        final_aligned_score = example.alignment_matrix.max(axis=-1).mean()
        example.final_aligned_score = final_aligned_score
    
    print("----- Saving results -----")
    data_to_save = [example.to_dict() for example in politeness_examples]
    with open("../results/{}/politeness_{}.json".format(baseline, model), "w") as f:
        json.dump(data_to_save, f, indent=4)


if __name__ == "__main__":
    politeness_data = load_politeness_data()

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="vanilla")
    parser.add_argument("--model", type=str, default="gpt-5.2-pro-2025-12-11")
    args = parser.parse_args()
    baseline = args.baseline
    model = args.model
    assert baseline in ["vanilla", "cot", "socratic", "subq"]
    assert model in ["gpt-5.2-pro-2025-12-11", "gpt-5-mini-2025-08-07", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001", "gemini-2.5-pro", "gemini-2.5-flash"]

    #set cache directory
    cache = Cache("/shared_data0/shreyah/llm_cache/politeness/{}".format(baseline))

    run_pipeline(politeness_data, baseline=baseline, model=model)
