import pandas as pd
import numpy as np
from datasets import load_dataset
import openai
from openai import OpenAI
import json
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
        self.mse = 0.0
        self.claims = []
        self.relevant_claims = []
        self.alignment_scores = []
        self.alignment_categories = []
        self.alignment_reasonings = []
        self.final_alignment_score = 0.0
    
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
            'alignment_scores': self.alignment_scores,
            'alignment_categories': self.alignment_categories,
            'alignment_reasonings': self.alignment_reasonings,
            'final_alignment_score': self.final_alignment_score
        }


@cache.memoize()
def query_openai(prompt, model="gpt-4o"):
    """
    Sends a prompt to the OpenAI chat API and returns the model's response.

    Uses diskcache to memoize results to avoid repeated API calls.
    Handles up to three retries in case of errors.

    Args:
        prompt (str): The input prompt to send to the model.
        model (str): The OpenAI model to use (default is "gpt-4o").

    Returns:
        str: The content of the model's response, or "ERROR" if all retries fail.
    """
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
    response = query_openai(prompt).replace("\n\n", "\n")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
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

def calculate_expert_alignment_score(claim: str):
    """
    Assesses a claim's alignment with expert criteria using the alignment prompt.

    Extracts the category, numeric alignment score, and reasoning.

    Args:
        claim (str): A relevant claim to evaluate.

    Returns:
        Tuple[str, float, str] or None: The alignment category, score, and reasoning, or None on error.
    """
    prompt = alignment_politeness.format(claim)
    response = query_openai(prompt).replace("\n\n", "\n")
    if response == "ERROR":
        print("Error in querying OpenAI API")
        return None
    response = response.replace("Category:", "").strip()
    response = response.split("\n")
    category = response[0].strip()
    alignment_score = response[1].replace("Category Alignment Rating:", "").strip()
    reasoning = response[2].replace("Reasoning:", "").strip()
    try:
        alignment_score = float(alignment_score)
        assert(len(category) > 5)
        assert(len(reasoning) > 10)
    except:
        print("ERROR: Could not convert alignment score to float")
        print(response)
        alignment_score = 0.0
    return category, alignment_score, reasoning

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
            politeness_data = politeness_data.sample(6, random_state=11).reset_index(drop=True)
            sampled_data = pd.concat([sampled_data, politeness_data], ignore_index=True)
    sampled_data = sampled_data.reset_index(drop=True)
    return sampled_data


def run_pipeline(politeness_data, baseline="vanilla"):
    """
    Executes the full politeness evaluation pipeline on a dataset of utterances.

    The pipeline consists of:
    1. Generating LLM scores and explanations.
    2. Extracting claims from explanations.
    3. Filtering relevant claims.
    4. Scoring alignment of claims using expert heuristics.
    5. Saving the resulting list of annotated PolitenessExample objects to disk.

    Args:
        politeness_data (pandas.DataFrame): The input dataset with columns including 'Utterance' and 'politeness'.
        baseline (str): The prompting baseline strategy to use (e.g., "vanilla", "cot").
    
    Returns:
        None: Results are saved as a JSON file under `../results/{baseline}/politeness.json`.
    """
    politeness_examples = []
    for idx,row in tqdm(politeness_data.iterrows()):
        rating, explanation = get_llm_generated_answer(row['Utterance'], baseline)
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

    for example in politeness_examples:
        claims = isolate_individual_features(example.llm_explanation)
        if claims is None:
            continue
        example.claims = [claim.strip() for claim in claims]

    for example in politeness_examples:
        relevant_claims = distill_relevant_features(example)
        example.relevant_claims = relevant_claims

    for example in politeness_examples:
        alignment_scores = []
        alignment_categories = []
        alignment_reasonings = []
        for claim in tqdm(example.relevant_claims):
            category, alignment_score, reasoning = calculate_expert_alignment_score(claim)
            if category is None:
                continue
            alignment_scores.append(alignment_score)
            alignment_categories.append(category)
            alignment_reasonings.append(reasoning)
        example.alignment_scores = alignment_scores
        example.alignment_categories = alignment_categories
        example.final_alignment_score = np.sum(alignment_scores)/len(example.claims)
        example.alignment_reasonings = alignment_reasonings

        
    data_to_save = [example.to_dict() for example in politeness_examples]
    with open("../results/{}/politeness_gpt-4o.json".format(baseline), 'w') as f:
        json.dump(data_to_save, f, indent=4)

if __name__ == "__main__":
    politeness_data = load_politeness_data()
    run_pipeline(politeness_data, baseline="vanilla")
    run_pipeline(politeness_data, baseline="cot")
    run_pipeline(politeness_data, baseline="socratic")
    run_pipeline(politeness_data, baseline="subq")