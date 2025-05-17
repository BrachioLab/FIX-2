from typing import Dict, List, Tuple, Any, Union, Callable
import torch
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import wfdb
from wfdb import Record
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm
from datasets import load_dataset
import json

from llms import load_model
import humanize
import openai
from diskcache import Cache

from prompts.explanations import cardiac_prompt, vanilla_baseline, cot_baseline, socratic_baseline, least_to_most_baseline
from prompts.claim_decomposition import decomposition_cardiac
from prompts.relevance_filtering import relevance_cardiac, load_relevance_cardiac_prompt
from prompts.expert_alignment import alignment_cardiac

with open("../OPENAI_API_KEY.txt", "r") as file:
    openai_api_key = file.read()
with open("../ANTHROPIC_API_KEY.txt", "r") as file:
    anthropic_api_key = file.read()
with open("../GOOGLE_API_KEY.txt", "r") as file:
    google_api_key = file.read()


ROOT_DIR = '..'
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
os.environ['GOOGLE_API_KEY'] = google_api_key
os.environ['CACHE_DIR'] = os.path.join(ROOT_DIR, 'cache_dir')

client = openai.OpenAI(api_key=openai.api_key)
cache = Cache(os.environ.get("CACHE_DIR"))

default_model = "gpt-4o"


# **Task:** 
# Given a patient's age, gender, race, ICU visit reason, and 2 minutes of ECG data at 500 Hz (in a graph image), 
# predict whether this patient gets cardiac arrest in the next 5 minutes.
#
# [set the corresponding parameters below:]
DURATION_SEC = 120
PRED_WINDOW_SEC = 300
# FS = 500


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return clean_for_json(obj.to_dict())
    elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    elif isinstance(obj, pd.NaT.__class__) or pd.isna(obj):
        return None
    else:
        return obj
        
class CardiacExample:
    def __init__(self,
                 data,
                 ground_truth: Any,
                 llm_label: Any,
                 llm_explanation: str,
                 duration_sec = DURATION_SEC,
                 pred_window_sec = PRED_WINDOW_SEC
                ):
        self.data = clean_for_json(data)
        self.data['p_signal'] = [e.item() for e in self.data['p_signal']]
        self.background = f"The patient is age {data['Age']}, gender {data['Gender']}, race {data['Race']}, and was admitted to the ICU for {data['Dx_name']}."
        self.duration_sec = duration_sec
        self.pred_window_sec = pred_window_sec
        
        self.ground_truth = ground_truth
        self.llm_label = llm_label 
        self.llm_explanation = llm_explanation

        # All raw claims obtained from the LLM
        self.all_claims : list[str] = []

        # Claims that are relevant to the explanation
        self.relevant_claims : list[str] = []

        # Relevant claims for which the LLM successfully managed to make an alignment judgment.
        self.alignable_claims : list[str] = []
        # self.alignment_categories : list[int] = [] # Same length as alignable claims
        self.alignment_category_ids : list[int] = [] # Same length as alignable claims
        self.alignment_scores : list[float] = [] # Same length as alignable claims
        self.alignment_reasonings : list[str] = [] # Same length as alignable claims

        # The final alignment score, computed as the mean of the alignment scores of the alignable claims.
        self.final_alignment_score : float = 0.0

        self.accuracy = int(self.ground_truth == self.llm_label)


    def to_dict(self):
        return {
            # "data": self.data,
            # "selected_data": {'record_name': self.data['record_name'], 
            #                   'n_sig': self.data['n_sig'],
            #                   'fs': self.data['fs'],
            #                   'age': self.data['age'],
            #                   'p_signal': self.
            #                  }
            "record_name": self.data['record_name'],
            "background": self.background,
            "duration_sec": self.duration_sec,
            "pred_window_sec": self.pred_window_sec,
            "ground_truth": self.ground_truth,
            "llm_label": self.llm_label,
            "llm_explanation": self.llm_explanation,
            "all_claims": self.all_claims,
            "relevant_claims": self.relevant_claims,
            "alignable_claims": self.alignable_claims,
            "alignment_category_ids": self.alignment_category_ids,
            # "alignment_categories": self.alignment_categories,
            "alignment_scores": self.alignment_scores,
            "alignment_reasonings": self.alignment_reasonings,
            "final_alignment_score": self.final_alignment_score,
            "accuracy": self.accuracy
        }



def cardiac_ecg_to_pil(record_dict, dpi=200) -> PIL.Image:
    record_name = record_dict['record_name']
    image_path = f'../notebooks/_dump/cardiac/images/{record_name}.png'
    if not os.path.exists(image_path):
        record = Record(**record_dict)
        wfdb.plot_wfdb(record, return_fig=True)
        plt.savefig(image_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    image = PIL.Image.open(image_path)
    image.load()
    return image


@cache.memoize()
def get_llm_output(prompt, images=None, model=default_model):
    """
    prompt: str
    images: list of PIL images
    system_prompt: str
    """
    llm = load_model(model)
    result = llm([(prompt, *images)])[0]
    return result



def get_llm_generated_answer(
    example_data: dict(),
    method: str = "vanilla",
    model: str = default_model,
    cardiac_ecg_to_pil: Callable = cardiac_ecg_to_pil,
    duration_sec = DURATION_SEC,
    pred_window_sec = PRED_WINDOW_SEC
) -> str:
    """
    Args:
        example : Cardiac Example data input 
        (str  Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the emotion classification task.
    """

    if method == 'least_to_most':
        method = 'subq'

    if method == "vanilla":
        prompt = cardiac_prompt.replace("[BASELINE_PROMPT]", '')
    elif method == "cot":
        prompt = cardiac_prompt.replace("[BASELINE_PROMPT]", cot_baseline)
    elif method == "socratic":
        prompt = cardiac_prompt.replace("[BASELINE_PROMPT]", socratic_baseline)
    elif method == "subq":
        prompt = cardiac_prompt.replace("[BASELINE_PROMPT]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid method: {method}")

    duration_str = humanize.precisedelta(duration_sec)
    pred_window_str = humanize.precisedelta(pred_window_sec)
    background_str = f"The patient is age {example_data['Age']}, gender {example_data['Gender']}, race {example_data['Race']}, and was admitted to the ICU for {example_data['Dx_name']}."
    prompt = prompt.format(duration_str, example_data['fs'], pred_window_str, background_str)

    # prompt = prompt.replace(
    #     '[LAST_IMAGE_NUM]',
    #     '1'
    # )

    record_keys = ['record_name', 'n_sig', 'fs', 'counter_freq', 'base_counter', 'sig_len', 'base_time', 'base_date', 'comments', 'sig_name', 'p_signal', 'd_signal', 'e_p_signal', 'e_d_signal', 'file_name', 'fmt', 'samps_per_frame', 'skew', 'byte_offset', 'adc_gain', 'baseline', 'units', 'adc_res', 'adc_zero', 'init_value', 'checksum', 'block_size']
    record_dict = {k: v for k, v in example_data.items() if k in record_keys}
    record_dict['p_signal'] = np.array(record_dict['p_signal']).reshape(-1, 1) 
    # record_dict['p_signal'] = example_data['p_signal'].reshape(-1, 1)
    image_pil = [cardiac_ecg_to_pil(record_dict)]

    llm_response = get_llm_output(prompt, image_pil, model=model)

    try:
        response_split = [r.strip().replace("**", "") for r in llm_response.split("\n") if r.strip() != "" \
        and r.strip().replace("**", "").startswith("Explanation:") or r.strip().replace("**", "").startswith("Prediction:")]
    
        llm_answer = response_split[0].split("Prediction: ")[1].strip().strip("*")
        explanation = response_split[1].split("Explanation: ")[1].strip().strip("*")
        return llm_answer, explanation
    except Exception as e:
        print(f"Error in parsing response {llm_response}")
        # import pdb; pdb.set_trace()
        return None, None


def isolate_individual_features(
    example: CardiacExample,
    # explanation: str | list[str],
    model: str = "gpt-4o",
    # duration_sec = DURATION_SEC,
    # pred_window_sec = PRED_WINDOW_SEC,
    # fs = FS
) -> list[str]:
    """
    Isolate individual features from the explanation by breaking it down into atomic claims.

    Args:
        # explanation (str): The explanation text to break down into claims
        example : Cardiac Example
        model (str): The OpenAI model to use for processing

    Returns:
        list[str]: A list of atomic claims extracted from the explanation
    """

    llm = load_model(model)

    duration_str = humanize.precisedelta(example.duration_sec)
    pred_window_str = humanize.precisedelta(example.pred_window_sec)
    fs = example.data['fs']
    explanation = example.llm_explanation

    if isinstance(explanation, list):
        prompts = [decomposition_cardiac.format(pred_window_str, fs, duration_str, e) for e in explanation]
        results = llm(prompts)
        all_all_claims: list[list[str]] = [
            [c.strip() for c in result.split("\n") if c.strip()]
            for result in results
        ]
        return all_all_claims
    else:
        raw_output = llm(decomposition_cardiac.format(pred_window_str, fs, duration_str, explanation))
        all_claims = [c.strip() for c in raw_output.split("\n") if c.strip()]
        all_claims = [c[2:] if c.startswith("- ") else c for c in all_claims]
        return all_claims


def distill_relevant_features(
    example: CardiacExample,
    # example_background: str,
    # example_image: PIL.Image.Image | torch.Tensor | np.ndarray,
    # answer: str,
    # atomic_claims: list[str],
    model: str = "gpt-4o",
    # verbose: bool = False,
) -> list[str]:
    """
    Distill the relevant features from the atomic claims.
    """

    atomic_claims = example.all_claims

    # assume that the example's ECG graph image is already there (because we used it for the generating prediction + explanation stage
    record_keys = ['record_name', 'n_sig', 'fs', 'counter_freq', 'base_counter', 'sig_len', 'base_time', 'base_date', 'comments', 'sig_name', 'p_signal', 'd_signal', 'e_p_signal', 'e_d_signal', 'file_name', 'fmt', 'samps_per_frame', 'skew', 'byte_offset', 'adc_gain', 'baseline', 'units', 'adc_res', 'adc_zero', 'init_value', 'checksum', 'block_size']
    record_dict = {k: v for k, v in example.data.items() if k in record_keys}
    # record_dict['p_signal'] = np.array(record_dict['p_signal']) 
    record_dict['p_signal'] = np.array(example.data['p_signal']).reshape(-1, 1)
    image = cardiac_ecg_to_pil(record_dict)

    prompts = [load_relevance_cardiac_prompt(
                    example.background, 
                    image, 
                    example.llm_label,
                    claim
                ) for claim in atomic_claims]

    llm = load_model(model)
    results = llm(prompts)


    relevant_claims = [
        claim for claim, result in zip(atomic_claims, results)
        if "relevance: yes" in result.lower()
    ]

    return relevant_claims


def calculate_expert_alignment_scores(
    claims: list[str],
    pred_window_sec=PRED_WINDOW_SEC, 
    model: str = "gpt-4o"
) -> list[dict]:
    """
    Computes the individual (and overall) alignment score of all the relevant claims.

    Args:
        claims (list[str]): A list of strings where each string is a relevant claim.
        model (str): The model to use for evaluation.

    Returns:
        dict: A dictionary containing:
            - alignment_scores: Mapping of each claim to its alignment score (1-5)
            - total_score: Overall alignment score across all claims
    """

    llm = load_model(model)

    pred_window_str = humanize.precisedelta(pred_window_sec)
    prompts = [alignment_cardiac.format(pred_window_str, claim) for claim in claims]
    responses = llm(prompts)

    results = []
    for i, response in enumerate(responses):
        clean_response = [s.strip() for s in response.split("\n") if s.strip()]
        try:
            if len(clean_response) == 3:
                # category = clean_response[0].split(": ")[1]
                category_id = clean_response[0].split(": ")[1]
                alignment = float(clean_response[1].split(": ")[1])
                reasoning = clean_response[2].split(": ")[1]
                
                results.append({
                    "Claim": claims[i],
                    # "Category": category,
                    "Category ID": category_id,
                    "Alignment": alignment,
                    "Reasoning": reasoning,
                })
            # if len(clean_response) == 4:
            #     category = clean_response[0].split(": ")[1]
            #     category_id = int(clean_response[1].split(": ")[1])
            #     alignment = float(clean_response[2].split(": ")[1])
            #     reasoning = clean_response[3].split(": ")[1]
                
            #     results.append({
            #         "Claim": claims[i],
            #         "Category": category,
            #         "Category ID": category_id,
            #         "Alignment": alignment,
            #         "Reasoning": reasoning,
            #     })

        except Exception as e:
            continue

    return results


def cardiac_data_to_examples(
    cardiac_data: pd.DataFrame,
    explanation_model: str = default_model,
    evaluation_model: str = default_model,
    baseline: str = "vanilla",
    verbose: bool = False,
) -> list[CardiacExample]:
    """
    The full LLM pipeline essentially - we are converting the cardiac data instances into a CardiacExamples.
    """
    _start_time = time.time()

    # Step 0: Get the LLM answers
    _t = time.time()

    cardiac_examples = []
    for idx, row in tqdm(cardiac_data.iterrows()):
        llm_label, explanation = get_llm_generated_answer(row, baseline, explanation_model)
        # print(llm_label, explanation)
        if llm_label is None:
            continue
        cardiac_examples.append(CardiacExample(
            data=row,
            ground_truth=row['label'],
            llm_label=llm_label,
            llm_explanation=explanation
        ))


    # Step 1: Decompose the LLM explanation into atomic claims
    _t = time.time()
    for example in tqdm(cardiac_examples):
        claims = isolate_individual_features(example, evaluation_model)
        if claims is None:
            continue
        example.all_claims = [claim.strip() for claim in claims]
    
    if verbose:
        print(f"Time taken to decompose into atomic claims: {time.time() - _t:.3f} seconds")

    # we should also save these just in case because we will use it in the latter parts
    for example in tqdm(cardiac_examples):
        torch.save(example, f"../notebooks/_dump/cardiac/final/gpt-4o/cardiac_examples/{example.data['record_name']}")

    
    # Step 2: Distill the relevant features from the atomic claims
    _t = time.time()
    for example in tqdm(cardiac_examples):
        example.relevant_claims = distill_relevant_features(example, evaluation_model)
    if verbose:
        print(f"Time taken to distill relevant features: {time.time() - _t:.3f} seconds")


    # Step 3: Calculate the expert alignment scores
    _t = time.time()

    for example in tqdm(cardiac_examples):
        align_infos = calculate_expert_alignment_scores(example.relevant_claims)
    
        example.alignable_claims = [info["Claim"] for info in align_infos]
        # example.alignment_categories = [info["Category"] for info in align_infos]
        example.alignment_category_ids = [info["Category ID"] for info in align_infos]
        example.alignment_scores = [info["Alignment"] for info in align_infos]
        example.alignment_reasonings = [info["Reasoning"] for info in align_infos]
        example.final_alignment_score = np.mean(example.alignment_scores)

        # Non-alignable claims are given a score of 0.0
        if len(align_infos) > 0:
            example.final_alignment_score = sum(score for score in example.alignment_scores) / len(example.all_claims)
        else:
            example.final_alignment_score = 0.0

    if verbose:
        print(f"Time taken to calculate expert alignment scores: {time.time() - _t:.3f} seconds")

    if verbose:
        print(f"Total time taken: {time.time() - _start_time:.3f} seconds")

    return cardiac_examples


def run_cardiac_pipeline(
    cardiac_data: pd.DataFrame,
    explanation_model: str = default_model,
    evaluation_model: str = default_model,
    baseline: str = "vanilla",
    verbose: bool = False,
    overwrite_existing: bool = False,
) -> list[CardiacExample]:
    """
    Run the cardiac pipeline on a pd.DataFrame of cardia data instances.
    """
    save_path = str(Path(__file__).parent / ".." / "results" / baseline / f"cardiac_{explanation_model}.json")
    if os.path.exists(save_path) and not overwrite_existing:
        print(f"Results already exist at {save_path}. Set overwrite_existing=True to overwrite.")
        return

    examples = cardiac_data_to_examples(cardiac_data, explanation_model, evaluation_model, baseline, verbose)

    with open(save_path, "w") as f:
        json.dump([example.to_dict() for example in examples], f, indent=4)


def get_yes_no_confirmation(prompt):
    """
    Prompts the user with a yes/no question and returns True for yes, False for no.
    Keeps asking until a valid response is given.
    """
    while True:
        response = input(prompt + " (Y/n): ").lower().strip()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")


if __name__ == "__main__":
    _start_time = time.time()

    # Take a few random, unique samples from the dataset
    random.seed(42)
    num_samples = 200 #120

    ds = load_dataset("BrachioLab/mcmed-cardiac")
    cardiac_data_all = ds['train'].to_pandas()
    cardiac_data = cardiac_data_all.sample(num_samples, random_state=11).reset_index(drop=True)
    cardiac_data['label'] = cardiac_data['label'].map({True: 'Yes', False: 'No'})

    # models = ["gpt-4o", "o1", "claude-3-5-sonnet-latest", "gemini-2.5-pro-exp-03-25"]
    # models = ["gpt-4o", "o1", "claude-3-5-sonnet-latest", "gemini-2.0-flash"]
    models = ["gpt-4o"]
    baselines = ["vanilla", "cot", "socratic", "subq"]

    # Can be very expensive!
    if get_yes_no_confirmation("You are about to spend a lot of money"):
        # Run the models and baselines
        for model in models:
            _model_time = time.time()
            for baseline in baselines:
                print(f"\nRunning {model} with {baseline} baseline...")
                run_cardiac_pipeline(
                    cardiac_data=cardiac_data,
                    explanation_model=model,
                    evaluation_model="gpt-4o",
                    baseline=baseline,
                    verbose=True,
                )
            print(f"Time taken for {model}: {time.time() - _model_time:.3f} seconds")

    else:
        print("Your bank account is safe!")

    print(f"Total time taken: {time.time() - _start_time:.3f} seconds")
